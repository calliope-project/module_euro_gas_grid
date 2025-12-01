rule cluster_gas_network:
    message: "Clustering and sectioning exisiting gas grid network for {wildcards.resolution} resolution"
    input:
        scigrid = "resources/automatic/gas_grid.zip",
        offshore_grid = "resources/automatic/gas_grid.zip",
        regions = "resources/user/shapes_{resolution}.geojson",
    output: "results/shapes/{resolution}/pipe_clusters.geojson",
    conda: "../envs/cluster.yaml"
    script: "../scripts/gas_network_clustering.py"


rule cluster_salt_cavern_potentials:
    message: "Clustering asalt_cavern_potenaials {wildcards.resolution} resolution"
    input:
        salt_cavern_potentials = "resources/user/shapes_salt_caverns_potential.geojson",
        regions = "resources/user/shapes_{resolution}.geojson",
    output:
        "results/shapes/{resolution}/salt_cavern.geojson",

    conda: "../envs/cluster.yaml"
    script: "../scripts/salt_cavern.py"

