import json

def get_model(algorithm_name, env, **kwargs):
    # Return the appropriate model based on name
    pass

def run_experiments(config_path):
    with open(config_path) as config_file:
        config = json.load(config_file)

    for experiment in config["experiments"]:
        env = make_vec_env(experiment["env"])
        model = get_modell(experimen["algorithm"], env, **experiment["params"])
        # Train the model and evaluate results
        print(f"Completed: {experiment['env']} with {experiment['algorithm']}")

    if __name__=="__main__":
        run_experiments('config.json')