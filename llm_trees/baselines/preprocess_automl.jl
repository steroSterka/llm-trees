using CategoricalArrays
using CSV
using DataFrames


function preprocess_automl(X, nominal_features, method)

	feature_inds = []
	one_hot_names_list = []
	
	if method == "autogluon"
		# categorical encoding (strings -> integers)
		for nominal_feature in nominal_features
			X[!, nominal_feature] .= levelcode.(CategoricalArray(X[!, nominal_feature]))
		end
		feature_inds = Dict(feature => if (feature in nominal_features) "category" else "float" end
			for feature in names(X))
	elseif method == "autoprognosis"
		# categorical encoding (strings -> integers)
		for nominal_feature in nominal_features
			X[!, nominal_feature] .= levelcode.(CategoricalArray(X[!, nominal_feature]))
		end
	elseif method == "tabpfn"
		# one-hot encoding
		for nominal_feature in nominal_features
			one_hot_names = nominal_feature .* "_" .* string.(unique(skipmissing(X[!, nominal_feature])))
			one_hot_values = permutedims(unique(X[!, nominal_feature])
				.== permutedims(X[!, nominal_feature]))
	
			one_hot_encoding = DataFrame([one_hot_names[i] => one_hot_values[:, i]
				for i in 1:length(one_hot_names)])
			X = hcat(X, one_hot_encoding)
			select!(X, Not(nominal_feature))
			append!(one_hot_names_list, one_hot_names)
		end
		feature_inds = [columnindex(X, one_hot_name) for one_hot_name in one_hot_names_list]
	end

	# convert all entries to float (except missing values)
	transform!(X, All() .=> ByRow(passmissing(Float64)), renamecols=false)

	return X, feature_inds

end
