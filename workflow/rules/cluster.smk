"""Clustering rules."""

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


rule cluster_gas_network:
    message: "Clustering and sectioning existing gas grid to {wildcards.shapes}."
    params:
        projected_crs = config["crs"]["projected"],
        replace_sovereign = config["clustering"].get("replace_sovereign", {})
    input:
        pipelines = rules.prepare_pipelines.output.pipelines,
        nodes = rules.prepare_pipelines.output.nodes,
        shapes = "resources/user/{shapes}/shapes.parquet",
    output:
        hubs = "results/{shapes}/hubs.parquet",
        pipelines = "results/{shapes}/pipelines.parquet",
        nodes = "results/{shapes}/nodes.parquet",
        fig = "results/{shapes}/pipelines.png"
    log:
        "logs/{shapes}/cluster_gas_network.log"
    conda: "../envs/euro_gas_grid.yaml"
    script: "../scripts/cluster_gas_network.py"
