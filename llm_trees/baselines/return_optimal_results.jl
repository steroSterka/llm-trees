include("preprocess_optimal.jl")
include("return_balanced_accuracy.jl")
include("return_f1.jl")

using CSV
using DataFrames
using JuMP
using Gurobi
using PyCall


function return_optimal_results(filepath_in, filepath_out, fraction, eval, specifier="")

    results = DataFrame(dataset = [], method = [], iter = [], split = [], accuracy = [], f1_score = [])

    for method in ["bss", "oct"]
        for i = 1:5

            # read data
            data = split(filepath_in, "/")[end]
            X = CSV.read("$(filepath_in)/X.csv", DataFrame)
            y = CSV.read("$(filepath_in)/y.csv", DataFrame)

            # define shots
            if fraction < 1
                shot = floor(Int, fraction*nrow(X))
                splits = string(Int(fraction * 100)) * "/" * string(Int(100 - fraction * 100))
            else
                shot = Int(fraction)
                splits = string(Int(fraction)) * "-shot"
            end
            
            # preprocess data
            X_train, X_test, y_train, y_test = preprocess_optimal(X, y, data, shot, i, eval)
            y_train = y_train[!, 1]
            y_test = y_test[!, 1]

            if method == "bss"
                # optimal feature selection for classification
                # (https://link.springer.com/article/10.1007/s10994-021-06085-5)
                grid = IAI.GridSearch(
                    IAI.OptimalFeatureSelectionClassifier(
                        # features are "min-max" scaled by default
                        relaxation=false,
                        solver=optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0),
                        random_seed=i,
                    ),
                    # sparsity=1:4 means that at most 4 features are selected
                    # (https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2795657)
                    sparsity=1:4,
                    # https://link.springer.com/chapter/10.1007/978-3-031-42608-7_10
                    gamma=[0.1, 0.02, 0.004],
                )
            elseif method == "oct"
                # optimal classification tree
                # (https://link.springer.com/article/10.1007/s10994-017-5633-9)
                grid = IAI.GridSearch(
                    IAI.OptimalTreeClassifier(
                        random_seed=i,
                    ),
                    # max_depth=1:2 means that at most 3 features are selected
                    max_depth=1:2,
                )
            end

            try
                if nrow(unique(y)) == 2
                    IAI.fit_cv!(grid, X_train, y_train, n_folds=3, validation_criterion=:f1score,
                        positive_label=1)
                else
                    IAI.fit_cv!(grid, X_train, y_train, n_folds=3)
                end
                push!(results, [replace(data, "analcatdata_" => ""), method, i - 1, splits,
                    py"return_balanced_accuracy"(y_test, IAI.predict(grid, X_test)),
                    py"return_f1"(y_test, IAI.predict(grid, X_test))])
            catch
                continue
            end 
                
        end
    end

    mkpath("$(filepath_out)")
    if !isfile("$(filepath_out)/optimal_$(specifier)$(eval)_results.csv")
        CSV.write("$(filepath_out)/optimal_$(specifier)$(eval)_results.csv", results)
    else
        CSV.write("$(filepath_out)/optimal_$(specifier)$(eval)_results.csv", results, append=true)
    end

end
