import argparse
from llm_trees.llms import generate_gpt_tree, generate_claude_tree, generate_gemini_tree
from llm_trees.config import Config
from llm_trees.induction_utils import eval_induction
from llm_trees.embeddings import eval_embedding

def main():
    parser = argparse.ArgumentParser(description="LLM Trees: Decision Trees with Language Models")
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Subparser for the generate command
    parser_generate = subparsers.add_parser('generate', help='Generate decision trees')
    parser_generate.add_argument('--root', type=str, default=".", help="Root directory for the project")
    parser_generate.add_argument('--dataset', type=str, default="penguins", help="Dataset name")
    parser_generate.add_argument('--method', type=str, required=True, choices=['gpt-4o', 'gpt-o1', 'gemini', 'claude'], help="The LLM method to use (gpt, gemini, claude)")
    parser_generate.add_argument('--temperature', type=float, default=1, help="Temperature for the LLM")
    parser_generate.add_argument('--iter', type=int, default=-1, help="Iteration counter")
    parser_generate.add_argument('--num_iters', type=int, default=5, help="Number of iterations")
    parser_generate.add_argument('--train_split', type=float, default=0.66, help="Train/test split ratio")
    parser_generate.add_argument('--append_raw_features', action='store_true', help="Append raw features to the embeddings")
    parser_generate.add_argument('--force_decision_tree', action='store_true', help="Force the generation of a decision tree")
    parser_generate.add_argument('--include_description', action='store_true', help="Include feature descriptions of the dataset in the prompt")
    parser_generate.add_argument('--llm_dialogue', action='store_true', help="Enable LLM dialogue mode")
    parser_generate.add_argument('--max_tree_depth', type=int, default=2, help="Maximum depth of the decision tree")
    parser_generate.add_argument('--num_examples', type=int, default=3, help="Number of examples to provide in the prompt")
    parser_generate.add_argument('--num_retry_llm', type=int, default=10, help="Number of retries for generating a valid tree")
    parser_generate.add_argument('--use_role_prompt', action='store_true', help="Use role-based prompts for the LLM")
    parser_generate.add_argument('--num_trees', type=int, default=5, help="Number of trees")
    parser_generate.add_argument('--seed', type=int, default=42, help="Random seed")
    parser_generate.add_argument('--generate_tree_if_missing', action='store_true', help="Generate tree if missing")
    parser_generate.add_argument('--regenerating_invalid_trees', action='store_true', help="Regenerate invalid trees")
    parser_generate.add_argument('--skip_existing', action='store_true', help="Skip existing trees")

    # Subparser for the eval_induction command
    parser_eval_induction = subparsers.add_parser('eval_induction', help='Evaluate induction')
    parser_eval_induction.add_argument('--root', type=str, default=".", help="Root directory for the project")
    parser_eval_induction.add_argument('--dataset', type=str, default="penguins", help="Dataset name")
    parser_eval_induction.add_argument('--method', type=str, required=True, choices=['gpt-4o', 'gpt-o1', 'gemini', 'claude'], help="The LLM method to use (gpt, gemini, claude)")
    parser_eval_induction.add_argument('--temperature', type=float, default=1, help="Temperature for the LLM")
    parser_eval_induction.add_argument('--iter', type=int, default=-1, help="Iteration counter")
    parser_eval_induction.add_argument('--num_iters', type=int, default=5, help="Number of iterations")
    parser_eval_induction.add_argument('--train_split', type=float, default=0.66, help="Train/test split ratio")
    parser_eval_induction.add_argument('--force_decision_tree', action='store_true', help="Force the generation of a decision tree")
    parser_eval_induction.add_argument('--include_description', action='store_true', help="Include feature descriptions of the dataset in the prompt")
    parser_eval_induction.add_argument('--llm_dialogue', action='store_true', help="Enable LLM dialogue mode")
    parser_eval_induction.add_argument('--max_tree_depth', type=int, default=2, help="Maximum depth of the decision tree")
    parser_eval_induction.add_argument('--num_examples', type=int, default=3, help="Number of examples to provide in the prompt")
    parser_eval_induction.add_argument('--num_retry_llm', type=int, default=10, help="Number of retries for generating a valid tree")
    parser_eval_induction.add_argument('--use_role_prompt', action='store_true', help="Use role-based prompts for the LLM")
    parser_eval_induction.add_argument('--num_trees', type=int, default=5, help="Number of trees")
    parser_eval_induction.add_argument('--seed', type=int, default=42, help="Random seed")
    parser_eval_induction.add_argument('--generate_tree_if_missing', action='store_true', help="Generate tree if missing")
    parser_eval_induction.add_argument('--regenerating_invalid_trees', action='store_true', help="Regenerate invalid trees")
    parser_eval_induction.add_argument('--skip_existing', action='store_true', help="Skip existing trees")

    # Subparser for the eval_embedding command
    parser_eval_embedding = subparsers.add_parser('eval_embedding', help='Evaluate embedding')
    parser_eval_embedding.add_argument('--root', type=str, default=".", help="Root directory for the project")
    parser_eval_embedding.add_argument('--dataset', type=str, default="penguins", help="Dataset name")
    parser_eval_embedding.add_argument('--method', type=str, required=True, choices=['gpt-4o', 'gpt-o1', 'gemini', 'claude'], help="The LLM method to use (gpt, gemini, claude)")
    parser_eval_embedding.add_argument('--temperature', type=float, default=1, help="Temperature for the LLM")
    parser_eval_embedding.add_argument('--iter', type=int, default=-1, help="Iteration counter")
    parser_eval_embedding.add_argument('--num_iters', type=int, default=5, help="Number of iterations")
    parser_eval_embedding.add_argument('--train_split', type=float, default=0.66, help="Train/test split ratio")
    parser_eval_embedding.add_argument('--append_raw_features', action='store_true', help="Append raw features to the embeddings")
    parser_eval_embedding.add_argument('--force_decision_tree', action='store_true', help="Force the generation of a decision tree")
    parser_eval_embedding.add_argument('--include_description', action='store_true', help="Include feature descriptions of the dataset in the prompt")
    parser_eval_embedding.add_argument('--llm_dialogue', action='store_true', help="Enable LLM dialogue mode")
    parser_eval_embedding.add_argument('--max_tree_depth', type=int, default=2, help="Maximum depth of the decision tree")
    parser_eval_embedding.add_argument('--num_examples', type=int, default=3, help="Number of examples to provide in the prompt")
    parser_eval_embedding.add_argument('--num_retry_llm', type=int, default=10, help="Number of retries for generating a valid tree")
    parser_eval_embedding.add_argument('--use_role_prompt', action='store_true', help="Use role-based prompts for the LLM")
    parser_eval_embedding.add_argument('--num_trees', type=int, default=5, help="Number of trees")
    parser_eval_embedding.add_argument('--seed', type=int, default=42, help="Random seed")
    parser_eval_embedding.add_argument('--generate_tree_if_missing', action='store_true', help="Generate tree if missing")
    parser_eval_embedding.add_argument('--regenerating_invalid_trees', action='store_true', help="Regenerate invalid trees")
    parser_eval_embedding.add_argument('--skip_existing', action='store_true', help="Skip existing trees")

    args = parser.parse_args()

    config = Config(
        root=args.root,
        dataset=args.dataset,
        method=args.method,
        temperature=args.temperature,
        iter=args.iter,
        num_iters=args.num_iters,
        train_split=args.train_split,
        append_raw_features=args.append_raw_features,
        force_decision_tree=args.force_decision_tree,
        include_description=args.include_description,
        llm_dialogue=args.llm_dialogue,
        max_tree_depth=args.max_tree_depth,
        num_examples=args.num_examples,
        num_retry_llm=args.num_retry_llm,
        use_role_prompt=args.use_role_prompt,
        num_trees=args.num_trees,
        seed=args.seed,
        generate_tree_if_missing=args.generate_tree_if_missing,
        regenerating_invalid_trees=args.regenerating_invalid_trees,
        skip_existing=args.skip_existing
    )

    if args.command == 'generate':
        if args.method == 'gpt':
            result = generate_gpt_tree(config)
        elif args.method == 'claude':
            result = generate_claude_tree(config)
        elif args.method == 'gemini':
            result = generate_gemini_tree(config)
        else:
            raise ValueError(f"Unknown method: {args.method}")
        print(result)
    elif args.command == 'eval_induction':
        accuracy, f1_score = eval_induction(config)
        print(f"Induction Evaluation - Accuracy: {accuracy}, F1 Score: {f1_score}")
    elif args.command == 'eval_embedding':
        accuracy, f1_score = eval_embedding(config)
        print(f"Embedding Evaluation - Accuracy: {accuracy}, F1 Score: {f1_score}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
