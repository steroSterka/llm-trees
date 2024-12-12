include("predict_automl.jl")
include("preprocess_automl.jl")
include("return_balanced_accuracy.jl")
include("return_f1.jl")

using CSV
using DataFrames
using Random


function return_automl_results(filepath_in, filepath_out, method, iter, fraction, eval, specifier="")
    
    results = DataFrame(dataset = [], method = [], iter = [], split = [], accuracy = [], f1_score = [])

    # read data
    data = split(filepath_in, "/")[end]
    X = CSV.read("$(filepath_in)/X.csv", DataFrame)
    y = CSV.read("$(filepath_in)/y.csv", DataFrame)

	# define nominal features and labels
	if data == "acl"
		nominal_features = ["Group", "Sex", "Dominant_Leg"]
        y[y[!, 1] .== -1, :] .= 0
	elseif data == "posttrauma"
		nominal_features = ["gender_birth", "ethnic_group", "education_age", "working_at_baseline",
			"penetrating_injury"]
        y[y[!, 1] .== -1, :] .= 0
    else
        nominal_features = []
	end
    y = y[!, 1]

    # define shots
    if fraction < 1
        shot = floor(Int, fraction*nrow(X))
        splits = string(Int(fraction * 100)) * "/" * string(Int(100 - fraction * 100))
    else
        shot = Int(fraction)
        splits = string(Int(fraction)) * "-shot"
    end
    
    # preprocess data
	X_preproc, feature_inds = preprocess_automl(deepcopy(X), nominal_features, method)
    train_X = X_preproc[shuffle(MersenneTwister(iter), 1:nrow(X))[1:shot], :]
    train_y = y[shuffle(MersenneTwister(iter), 1:nrow(X))[1:shot]]
    if eval == "train"
        test_X = X_preproc[shuffle(MersenneTwister(iter), 1:nrow(X))[1:shot], :]
        test_y = y[shuffle(MersenneTwister(iter), 1:nrow(X))[1:shot]]
    elseif eval == "test"
        test_X = X_preproc[shuffle(MersenneTwister(iter), 1:nrow(X))[shot+1:end], :]
        test_y = y[shuffle(MersenneTwister(iter), 1:nrow(X))[shot+1:end]]
    end

    # prediction
    try
        eval_y = py"predict_automl"(
            Array{Union{Missing, Float64}}(train_X),
            Array{Float64}(train_y),
            Array{Union{Missing, Float64}}(test_X),
            Array{Float64}(test_y),
            names(X_preproc),
            feature_inds,
            method
        )
        push!(results, [replace(data, "analcatdata_" => ""), method, iter - 1, splits,
            py"return_balanced_accuracy"(test_y, eval_y),
            py"return_f1"(test_y, eval_y)])
    catch
    end

    mkpath("$(filepath_out)")
    if !isfile("$(filepath_out)/automl_$(specifier)$(eval)_results.csv")
        CSV.write("$(filepath_out)/automl_$(specifier)$(eval)_results.csv", results)
    else
        CSV.write("$(filepath_out)/automl_$(specifier)$(eval)_results.csv", results, append=true)
    end
	
	return

end
