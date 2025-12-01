rule cluster_gas_network:
    message: "Clustering and sectioning exisiting gas grid network for {wildcards.shapes} resolution"
    input:
        scigrid = rules.prepare_pipelines.output.pipelines,
        regions = "resources/user/{shapes}/shapes.geojson",
    output:
        clusters="results/{shapes}/pipe_clusters.geojson",
    log:
        "logs/{shapes}/cluster_gas_network.log"
    conda: "../envs/euro_gas_grid.yaml"
    script: "../scripts/gas_network_clustering.py"


rule cluster_salt_cavern_potentials:
    message: "Clustering asalt_cavern_potenaials {wildcards.shapes} resolution"
    input:
        salt_cavern_potentials = "resources/user/salt_caverns_potential.geojson",
        regions = "resources/user/{shapes}/shapes.geojson",
    output:
        clusters="results/{shapes}/salt_cavern.geojson",
    log:
        "logs/{shapes}/cluster_salt_cavern_potentials.log"
    conda: "../envs/clustering.yaml"
    script: "../scripts/salt_cavern.py"

# rule cluster_existing_gas_network:
#     message: "Clustering existing gas network within {wildcards.shapes} resolution."
#     input:
#         pipelines = rules.prepare_pipelines.output.pipelines,
#         shapes = rules.
