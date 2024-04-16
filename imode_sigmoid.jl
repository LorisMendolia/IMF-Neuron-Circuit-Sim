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

function Imode_sigmoid_V(x,pars)

	@unpack Vin, Vlin, Vthr = pars

    V1, Vout, V2 = x

    [Icmp(Vin,x[1]) - I1(Vthr,x[1]) - Ilin1(x[1],x[2]),
    Ilin1(x[1],x[2]) - Ilin2(x[2],x[3]),
    Ilin2(x[2],x[3]) - Ilin3(Vlin,x[3])] ./C
    
end

V_P_diode(I) = Vdd - UT/κ * log(I/I0)
V_N_diode(I) = UT/κ * log(I/I0)

Iin_diode(Vin) = I0 * exp(κ*(Vdd-Vin)/UT)
Iout(Vout, Vgain) = I0 * exp((κ*Vout)/UT) / (1 + exp(κ*(Vout-Vgain)/UT))

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

	rfs(x, p) = (x2 = x[2], y=p) # Record the output voltage
	prob = BifurcationProblem(Imode_sigmoid_V, solNL.zero, pars_V, (@lens _.Vin), record_from_solution = rfs) # Set up the bifurcation problem with the input voltage as the bifurcation parameter

	# We set up continuation to use the selected input current range, by converting them to PMOS current mirror voltages
	opts = ContinuationPar(p_min = V_P_diode(Iin_range[2]), p_max = V_P_diode(Iin_range[1]), n_inversion = 50, ds = 1e-6, dsmin = 1e-12, dsmax = 1e-3, max_steps = 1000, nev = 3)
	br = continuation(prob, PALC(), opts; normC = norminf, bothside = true)

	if return_V
		return (br.branch.param, br.branch.x2)
	else
		return (Iin_diode.(br.branch.param), Iout.(br.branch.x2, Vgain))
	end

end

const memo_dict = Dict{NamedTuple, Interpolations.Extrapolation}()

function Imode_sigmoid_val(Iin, params)

	# Check if the model is already computed
	if haskey(memo_dict, params)
		sigmoid_int = memo_dict[params]
	else

		Irange = (0,1e-6)
		Iin_res, Iout_res = Imode_sigmoid_sim(Irange, params)

		if Iin_res[1] > Iin_res[end]
			Iin_res = reverse(Iin_res)
			Iout_res = reverse(Iout_res)
		end

		Interpolations.deduplicate_knots!(Iin_res, move_knots = true)
		Interpolations.deduplicate_knots!(Iout_res, move_knots = true)

		sigmoid_int = linear_interpolation(Iin_res, Iout_res, extrapolation_bc=Line());

		memo_dict[params] = sigmoid_int
	end

	return sigmoid_int(Iin)

end

function Imode_sigmoid_val_nomem(Iin, params)

	# @unpack Ithr, Igain, Ilin = params

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

export Imode_sigmoid_val, Imode_sigmoid_val_nomem

end