using CategoricalArrays
using CSV
using DataFrames
using Random


function preprocess_optimal(X, y, data, shot, seed, eval)

    try
        # few-shot learning
        X_train = X[shuffle(MersenneTwister(seed), 1:nrow(X))[1:shot], :]
        y_train = y[shuffle(MersenneTwister(seed), 1:nrow(X))[1:shot], :]
        if eval == "train"
            X_test = X[shuffle(MersenneTwister(seed), 1:nrow(X))[1:shot], :]
            y_test = y[shuffle(MersenneTwister(seed), 1:nrow(X))[1:shot], :]
        elseif eval == "test"
            X_test = X[shuffle(MersenneTwister(seed), 1:nrow(X))[shot+1:end], :]
            y_test = y[shuffle(MersenneTwister(seed), 1:nrow(X))[shot+1:end], :]
        end

        # define nominal and ordinal features
        if data == "acl"
            nominal_features = ["Group", "Sex", "Dominant_Leg"]
            ordinal_features = ["Tegner"]
        elseif data == "posttrauma"
            nominal_features = ["gender_birth", "ethnic_group", "education_age",
                "working_at_baseline", "penetrating_injury"]
            ordinal_features = ["smoker", "iss_category"]
        else
            nominal_features = []
            ordinal_features = []
        end

        # nominal feature encoding
        for nominal_feature in nominal_features
            X_train[!, nominal_feature] = CategoricalArray(X_train[!, nominal_feature])
            X_test[!, nominal_feature] = CategoricalArray(X_test[!, nominal_feature])
        end

        # missing value imputation
        # (https://www.jmlr.org/papers/v18/17-073.html)
        imputer = IAI.ImputationLearner(:opt_knn, random_seed=seed)
        X_train = IAI.fit_transform!(imputer, X_train)
        X_test = IAI.transform(imputer, X_test)
        
        # ordinal feature encoding
        for ordinal_feature in ordinal_features
            X_train[!, ordinal_feature] = CategoricalArray(X_train[!, ordinal_feature], ordered=true)
            X_test[!, ordinal_feature] = CategoricalArray(X_test[!, ordinal_feature], ordered=true)
        end

        return X_train, X_test, y_train, y_test

    catch
        preprocess_optimal(X, y, data, shot, seed+5, eval)
    end
    
end
