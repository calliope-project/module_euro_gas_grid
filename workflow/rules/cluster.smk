# rule cluster_gas_network:
#     message: "Clustering and sectioning exisiting gas grid network for {wildcards.shapes} resolution"
#     input:
#         scigrid = rules.prepare_pipelines.output.pipelines,
#         regions = "resources/user/{shapes}/shapes.geojson",
#     output:
#         clusters="results/{shapes}/pipe_clusters.geojson",
#     log:
#         "logs/{shapes}/cluster_gas_network.log"
#     conda: "../envs/euro_gas_grid.yaml"
#     script: "../scripts/gas_network_clustering.py"


# rule cluster_salt_cavern_potentials:
#     message: "Clustering asalt_cavern_potenaials {wildcards.shapes} resolution"
#     input:
#         salt_cavern_potentials = "resources/user/salt_caverns_potential.geojson",
#         regions = "resources/user/{shapes}/shapes.geojson",
#     output:
#         clusters="results/{shapes}/salt_cavern.geojson",
#     log:
#         "logs/{shapes}/cluster_salt_cavern_potentials.log"
#     conda: "../envs/clustering.yaml"
#     script: "../scripts/salt_cavern.py"

# rule cluster_existing_gas_network:
#     message: "Clustering existing gas network within {wildcards.shapes} resolution."
#     input:
#         pipelines = rules.prepare_pipelines.output.pipelines,
#         shapes = rules.

# rule cluster_and_snap_pipelines:
#     message:
#         "Clustering and snapping pipelines to nodes."
#     params:
#         buffer = config["imputation"]["buffer_distance"],
#         projected_crs = config["crs"]["projected"],
#     input:
#         pipelines=rules.prepare_pipelines.output.pipelines,
#         countries=rules.prepare_countries.output.countries,
#     output:
#         pipelines="resources/automatic/clustered/pipelines.parquet",
#         nodes="resources/automatic/clustered/nodes.parquet",
#         fig="resources/automatic/clustered/pipes_and_nodes.png"
#     log:
#         "logs/cluster_and_snap_pipelines.log",
#     conda:
#         "../envs/euro_gas_grid.yaml"
#     script:
#         "../scripts/cluster_and_snap_pipelines.py"

rule cluster_gas_network:
    message: "Clustering and sectioning existing gas grid to {wildcards.shapes}."
    params:
        projected_crs = config["crs"]["projected"],
        min_segment_length = config["clustering"]["min_segment_length"]

    input:
        countries = rules.prepare_countries.output.countries,
        pipelines = rules.prepare_pipelines.output.pipelines,
        nodes = rules.prepare_pipelines.output.nodes,
        shapes = "resources/user/{shapes}/shapes.parquet",
    output:
        pipelines = "results/{shapes}/pipelines.parquet",
        nodes = "results/{shapes}/nodes.parquet",
        fig = "results/{shapes}/pipelines.png"
    log:
        "logs/{shapes}/cluster_gas_network.log"
    conda: "../envs/euro_gas_grid.yaml"
    script: "../scripts/cluster_gas_network.py"
