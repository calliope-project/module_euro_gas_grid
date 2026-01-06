"""Clustering rules."""

rule cluster_gas_network:
    message: "Clustering and sectioning existing gas grid to {wildcards.shapes}."
    params:
        projected_crs = config["crs"]["projected"],
        replace_sovereign = config["clustering"]["pipelines"].get("replace_sovereign", {})
    input:
        pipelines = rules.prepare_pipelines.output.pipelines,
        nodes = rules.prepare_pipelines.output.nodes,
        shapes = "resources/user/{shapes}/shapes.parquet",
    output:
        hubs = "results/{shapes}/hubs.parquet",
        pipelines = "results/{shapes}/pipelines.parquet",
        nodes = "results/{shapes}/nodes.parquet",
        fig = report(
            "results/{shapes}/pipelines.png",
            caption="../report/cluster_gas_network.rst",
            category="Euro gas grid module",
        )
    log:
        "logs/{shapes}/cluster_gas_network.log"
    conda: "../envs/euro_gas_grid.yaml"
    script: "../scripts/cluster_gas_network.py"


rule cluster_salt_cavern_h2_potential:
    message: "Clustering of salt cavern H2 storage to {wildcards.shapes}."
    params:
        projected_crs = config["crs"]["projected"],
        min_gwh_tolerance = config["clustering"]["salt_caverns"]["min_gwh"]
    input:
        salt_caverns = rules.download_salt_cavern_storage.output.caverns,
        shapes = "resources/user/{shapes}/shapes.parquet",
    output:
        salt_cavern_h2_potential = "results/{shapes}/salt_cavern_h2_potential.parquet",
        fig = report(
            "results/{shapes}/salt_cavern_h2_potential.png",
            caption="../report/cluster_salt_cavern_h2_potential.rst",
            category="Euro gas grid module",
        )
    log:
        "logs/{shapes}/cluster_salt_cavern_h2_potential.log"
    conda: "../envs/euro_gas_grid.yaml"
    script: "../scripts/cluster_salt_cavern_h2_potential.py"
