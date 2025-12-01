rule visualize_output:
    message: "generting plots for pipe clusters and salt cavern potentials {wildcards.resolution} resolution"
    input:
        regions = "resources/user/shapes_{resolution}.geojson",
        salt_cavern = rules.cluster_salt_cavern_potentials.output[0],
        salt_cavern_potential = rules.cluster_salt_cavern_potentials.input.salt_cavern_potentials,
        pipe_clusters = rules.cluster_gas_network.output[0],
    output:
        salt_cavern = "results/figs/{resolution}/salt_cavern.svg",
        pipe_clusters = "results/figs/{resolution}/pipes.svg",
    params:
        plot_configs = config["plots"]
    conda: "../envs/plots.yaml"
    script: "../scripts/plots.py"
