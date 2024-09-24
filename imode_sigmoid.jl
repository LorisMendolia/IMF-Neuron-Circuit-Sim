#=
Current mode sigmoid circuit simulation module

The module exports the following function:
Imode_sigmoid_eval(Iin, params; var_gain = false, use_mem = true)

The function evaluates the output current of the sigmoid circuit for a given input current and parameters.
The parameters are a NamedTuple with the following fields:
- Ithr: The threshold current of the sigmoid
- Igain: The gain current of the sigmoid
- Ilin: The linear current of the sigmoid

The function has two optional arguments:
- var_gain: If true, the function will change its internal behavior to optimize memory usage when the gain current parameter frequently changes
- use_mem: If true, the function will use a memory to store already computed sigmoid models. This should be left true for normal operation, but can be set to false for debugging purposes

Author: Loris Mendolia
Date: 08/05/2024
University of Liège, Belgium
=#
module ImodeSigmoid

using BifurcationKit, Parameters, NLsolve, Interpolations
const BK = BifurcationKit

const I0 = 1e-12 # A
const κ = 0.7
const Vdd = 1.8 # V
const UT= 25*1e-3 # V
const C=1e-3 # F

# Static transistor equations
Icmp(Vin,V1) = I0 * exp(κ*(Vdd-Vin)/UT) * (1 - exp(-(Vdd-V1)/UT))
I1(Vthr,V1) = I0 * exp(κ*Vthr/UT) * (1 - exp(-V1/UT))
Ilin1(V1,Vout) = I0 * exp((κ*V1 - Vout)/UT) * (1 - exp(-(V1-Vout)/UT))
Ilin2(Vout,V2) = I0 * exp((κ*Vout - V2)/UT) * (1 - exp(-(Vout-V2)/UT))
Ilin3(Vlin,V2) = I0 * exp(κ*Vlin/UT) * (1 - exp(-V2/UT));

# Circuit "dynamic" equations
function Imode_sigmoid_V(x,pars)

	@unpack Vin, Vlin, Vthr = pars

    V1, Vout, V2 = x

    [Icmp(Vin,x[1]) - I1(Vthr,x[1]) - Ilin1(x[1],x[2]),
    Ilin1(x[1],x[2]) - Ilin2(x[2],x[3]),
    Ilin2(x[2],x[3]) - Ilin3(Vlin,x[3])] ./C
    
end

# Curent mirror diode current-voltage conversions
V_P_diode(I) = Vdd - UT/κ * log(I/I0)
V_N_diode(I) = UT/κ * log(I/I0)

# Input and output voltage to current conversions
Iin_diode(Vin) = I0 * exp(κ*(Vdd-Vin)/UT)
Iout(Vout, Vgain) = I0 * exp((κ*Vout)/UT) / (1 + exp(κ*(Vout-Vgain)/UT))

# Simulation of the sigmoid characteristic using BifurcationKit
function Imode_sigmoid_sim(Iin_range, params; return_V = false)

	@unpack Ithr, Igain, Ilin = params

	# Current mirror parameter conversion
	Vthr = V_N_diode.(Ithr)
	Vgain = V_N_diode(Igain)
	Vlin = V_N_diode.(Ilin)
	Vin = V_P_diode.(Igain) # We want to make an initial guess with a high current to avoid very small values for our variables

	pars_V = (Vin = Vin, Vlin = Vlin, Vthr = Vthr)

	x0=[1.7, 0.9, 0.3]

	solNL = nlsolve(x -> Imode_sigmoid_V(x,pars_V), x0, iterations=convert(Int64,1e6), ftol=1e-9, xtol=1e-6)

	rfs(x, p; k...) = (x2 = x[2], y=p) # Record the output voltage
	prob = BifurcationProblem(Imode_sigmoid_V, solNL.zero, pars_V, (@optic _.Vin), record_from_solution = rfs) # Set up the bifurcation problem with the input voltage as the bifurcation parameter

	# We set up continuation to use the selected input current range, by converting them to PMOS current mirror voltages
	opts = ContinuationPar(p_min = V_P_diode(Iin_range[2]), p_max = V_P_diode(Iin_range[1]), n_inversion = 50, ds = 1e-6, dsmin = 1e-12, dsmax = 1e-3, max_steps = 1000, nev = 3)
	br = continuation(prob, PALC(), opts; normC = norminf, bothside = true)

	# For optimization reasons, we either want to return the output voltage or convert it directly to the output current
	if return_V
		return (br.branch.param, br.branch.x2)
	else
		return (Iin_diode.(br.branch.param), Iout.(br.branch.x2, Vgain))
	end

end

# Storage for already computed sigmoid models
const memo_dict_Iout = Dict{NamedTuple, Interpolations.Extrapolation}()
const memo_dict_Vout = Dict{NamedTuple, Interpolations.Extrapolation}()

# Evaluate the output current of the sigmoid for a given input current
function Imode_sigmoid_eval_Iout(Iin, params)

	# Check if the model is already computed
	if haskey(memo_dict_Iout, params)
		sigmoid_int = memo_dict_Iout[params]
	else

		# Simulate the sigmoid for a given current range and interpolate
		# TODO: Make the current range adaptive, to make sure saturation is reached at the end of the simulation and a linear extrapolation is valid
		Irange = (0,1e-6)
		Iin_res, Iout_res = Imode_sigmoid_sim(Irange, params)

		if Iin_res[1] > Iin_res[end]
			Iin_res = reverse(Iin_res)
			Iout_res = reverse(Iout_res)
		end

		Interpolations.deduplicate_knots!(Iin_res, move_knots = true)
		Interpolations.deduplicate_knots!(Iout_res, move_knots = true)

		sigmoid_int = linear_interpolation(Iin_res, Iout_res, extrapolation_bc=Line()); # In the inactive and saturation regions, we extrapolate the response to a line

		memo_dict_Iout[params] = sigmoid_int
	end

	return sigmoid_int(Iin)

end

# Evaluate the output voltage of the sigmoid for a given input current
function Imode_sigmoid_eval_Vout(Iin, params)

	@unpack Ithr, Igain, Ilin = params

	# We do not treat Vgain as a parameter to avoid recomputing the model every time it changes
	Vgain = V_N_diode(Igain)
	params2 = (Ithr = Ithr, Ilin = Ilin)

	# Check if the model is already computed
	if haskey(memo_dict_Vout, params2)
		sigmoid_V_int = memo_dict_Vout[params2]
	else

		# Simulate the sigmoid for a given current range and interpolate
		# TODO: Make the current range adaptive, to make sure saturation is reached at the end of the simulation and a linear extrapolation is valid
		Irange = (0,1e-6)
		Vin_res, Vout_res = Imode_sigmoid_sim(Irange, params, return_V = true)

		Iin_res = Iin_diode.(Vin_res)

		if Iin_res[1] > Iin_res[end]
			Iin_res = reverse(Iin_res)
			Vout_res = reverse(Vout_res)
		end

		Interpolations.deduplicate_knots!(Iin_res, move_knots = true)
		Interpolations.deduplicate_knots!(Vout_res, move_knots = true)

		sigmoid_V_int = linear_interpolation(Iin_res, Vout_res, extrapolation_bc=Line()); # In the inactive and saturation regions, we extrapolate the response to a line

		memo_dict_Vout[params2] = sigmoid_V_int
	end

	return Iout(sigmoid_V_int(Iin), Vgain)
end

# For debugging purposes, same as Imode_sigmoid_eval_Iout but without memory
function Imode_sigmoid_eval_nomem(Iin, params)

	Irange = (0,1e-6)
	Iin_res, Iout_res = Imode_sigmoid_sim(Irange, params)

	if Iin_res[1] > Iin_res[end]
		Iin_res = reverse(Iin_res)
		Iout_res = reverse(Iout_res)
	end

	Interpolations.deduplicate_knots!(Iin_res, move_knots = true)
	Interpolations.deduplicate_knots!(Iout_res, move_knots = true)

	sigmoid_int = linear_interpolation(Iin_res, Iout_res, extrapolation_bc=Line());

	return sigmoid_int(Iin)

end

# Wrapper function to evaluate the sigmoid model
function Imode_sigmoid_eval(Iin, params; var_gain = false, use_mem = true)

	if use_mem
		if var_gain
			return Imode_sigmoid_eval_Vout(Iin, params)
		else
			return Imode_sigmoid_eval_Iout(Iin, params)
		end
	else
		return Imode_sigmoid_eval_nomem(Iin, params)
	end

end

export Imode_sigmoid_eval

end