using CSV
using CriticalDifferenceDiagrams
using DataFrames
using PGFPlots


columns = [:dataset, :method, :accuracy, :f1_score]

# induction loop
eval_results = DataFrame(
    [String15[], String15[], Int64[],  String15[], Float64[], Float64[]],
    [:dataset, :method, :iter, :split, :accuracy, :f1_score]
)
for method in ["induction", "optimal_test", "automl_test"]
    eval_results_single = CSV.read("results/$(method)_results.csv", DataFrame)
    append!(eval_results, eval_results_single)
end
eval_results_induction = DataFrame(
    [String15[], String15[], Float64[],  Float64[]],
    columns
)

for dataset in ["boxing1", "boxing2", "japansolvent", "colic", "heart_h", "hepatitis",
    "house_votes_84", "labor", "penguins", "vote", "bankruptcy", "creditscore", "irish",
    "acl", "posttrauma"]
    for sub_method in ["claude", "gemini", "gpt-4o", "gpt-o1", "bss", "oct", "autogluon",
        "autoprognosis", "tabpfn"]

        eval_results_induction_acc = eval_results[
            (eval_results[!, :dataset] .== dataset) .&
            (eval_results[!, :method] .== sub_method) .&
            (eval_results[!, :split] .== "67/33"), :accuracy
        ]
        eval_results_induction_f1 = eval_results[
            (eval_results[!, :dataset] .== dataset) .&
            (eval_results[!, :method] .== sub_method) .&
            (eval_results[!, :split] .== "67/33"), :f1_score
        ]
        eval_results_induction = vcat(eval_results_induction, DataFrame([
            repeat([dataset], 5),
            repeat([sub_method], 5),
            append!(eval_results_induction_acc, 0.0 for i=1:5-length(eval_results_induction_acc)),
            append!(eval_results_induction_f1, 0.0 for i=1:5-length(eval_results_induction_f1))
            ], columns)
        )

    end
end


# embedding loop
eval_results = CSV.read(
    "results/embedding_results.csv", DataFrame
)
eval_results_embedding = DataFrame(
    [String15[], String15[], Float64[],  Float64[]],
    columns
)

for dataset in ["boxing1", "boxing2", "japansolvent", "colic", "heart_h", "hepatitis",
    "house_votes_84", "labor", "penguins", "vote", "bankruptcy", "creditscore", "irish",
    "acl", "posttrauma"]
    for sub_method in ["no", "claude", "gemini", "gpt-4o", "gpt-o1", "rt-us", "et-ss",
        "rf-ss", "xg-ss", "et-sv", "rf-sv", "xg-sv"]

        eval_results_embedding_acc = eval_results[
            (eval_results[!, :dataset] .== dataset) .&
            (eval_results[!, :method] .== sub_method) .&
            (eval_results[!, :split] .== "67/33"), :accuracy
        ]
        eval_results_embedding_f1 = eval_results[
            (eval_results[!, :dataset] .== dataset) .&
            (eval_results[!, :method] .== sub_method) .&
            (eval_results[!, :split] .== "67/33"), :f1_score
        ]
        eval_results_embedding = vcat(eval_results_embedding, DataFrame([
            repeat([dataset], 5),
            repeat([sub_method], 5),
            append!(eval_results_embedding_acc, 0.0 for i=1:5-length(eval_results_embedding_acc)),
            append!(eval_results_embedding_f1, 0.0 for i=1:5-length(eval_results_embedding_f1))
            ], columns)
        )

    end
end


replace!(eval_results_induction.method, "claude" => "Claude 3.5 Sonnet")
replace!(eval_results_induction.method, "gemini" => "Gemini 1.5 Pro")
replace!(eval_results_induction.method, "gpt-4o" => "GPT-4o")
replace!(eval_results_induction.method, "gpt-o1" => "GPT-o1")
replace!(eval_results_induction.method, "bss" => "BSS")
replace!(eval_results_induction.method, "oct" => "OCTs")
replace!(eval_results_induction.method, "autogluon" => "AutoGluon")
replace!(eval_results_induction.method, "autoprognosis" => "AutoPrognosis")
replace!(eval_results_induction.method, "tabpfn" => "TabPFN")

replace!(eval_results_embedding.method, "no" => "No embedding")
replace!(eval_results_embedding.method, "claude" => "Claude 3.5 Sonnet")
replace!(eval_results_embedding.method, "gemini" => "Gemini 1.5 Pro")
replace!(eval_results_embedding.method, "gpt-4o" => "GPT-4o")
replace!(eval_results_embedding.method, "gpt-o1" => "GPT-o1")
replace!(eval_results_embedding.method, "rt-us" => "Random trees (unsuperv.)")
replace!(eval_results_embedding.method, "et-ss" => "Extra trees (self-superv.)")
replace!(eval_results_embedding.method, "rf-ss" => "Random forest (self-superv.)")
replace!(eval_results_embedding.method, "xg-ss" => "XGBoost (self-superv.)")
replace!(eval_results_embedding.method, "et-sv" => "Extra trees (superv.)")
replace!(eval_results_embedding.method, "rf-sv" => "Random forest (superv.)")
replace!(eval_results_embedding.method, "xg-sv" => "XGBoost (superv.)")


# critical difference diagrams
cdd_induction_acc = CriticalDifferenceDiagrams.plot(
    eval_results_induction,
    :method,
    :dataset,
    :accuracy,
    maximize_outcome=true
)
cdd_induction_f1 = CriticalDifferenceDiagrams.plot(
    eval_results_induction,
    :method,
    :dataset,
    :f1_score,
    maximize_outcome=true
)
cdd_embedding_acc = CriticalDifferenceDiagrams.plot(
    eval_results_embedding,
    :method,
    :dataset,
    :accuracy,
    maximize_outcome=true
)
cdd_embedding_f1 = CriticalDifferenceDiagrams.plot(
    eval_results_embedding,
    :method,
    :dataset,
    :f1_score,
    maximize_outcome=true
)


# helvetica
pushPGFPlotsPreamble("""
    \\usepackage{helvet}
""")


# save critical difference diagrams with missing runs replaced by 0.0!
PGFPlots.save("../../plots/cdd_induction_f1.pdf", cdd_induction_f1)
PGFPlots.save("../../plots/cdd_induction_f1.svg", cdd_induction_f1)
PGFPlots.save("../../plots/cdd_induction_acc.pdf", cdd_induction_acc)
PGFPlots.save("../../plots/cdd_induction_acc.svg", cdd_induction_acc)
PGFPlots.save("../../plots/cdd_embedding_f1.pdf", cdd_embedding_f1)
PGFPlots.save("../../plots/cdd_embedding_f1.svg", cdd_embedding_f1)
PGFPlots.save("../../plots/cdd_embedding_acc.pdf", cdd_embedding_acc)
PGFPlots.save("../../plots/cdd_embedding_acc.svg", cdd_embedding_acc)
