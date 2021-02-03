# Script to generate WCGNA network

library(WGCNA)
library("rjson")

json_data <- fromJSON(file='./Data/pipeline/human/network_analysis/config.json')
#Load expression df with normalized count and the network ID file
expression = read.table(file.path('./Data', json_data['normalized_counts']), header = TRUE)
network_IDs = read.csv('./Data/pipeline/human/network_analysis/network_IDs.csv', row.names = 1)

## Filter expression for network only expression
network_only_expression = expression[expression$id %in% network_IDs$X0,]

# Convert the id column to index and delete id column
rownames(network_only_expression) = network_only_expression$id
network_only_expression$id = NULL

## automatic block-wise network
# transpose the dataframe
network_only_expression_t = t(network_only_expression)

# blockwiseModules() will generate Tom network along with the module detection by WCGNA
net = blockwiseModules(network_only_expression_t, power = 14,
                       TOMType = "unsigned", minModuleSize = 100,
                       reassignThreshold = 0, detectCutHeight = 0.99,
                       numericLabels = TRUE, pamRespectsDendro = FALSE,
                       saveTOMs = F,
                       #   saveTOMFileBase = "femaleMouseTOM",
                       verbose = 3, deepSplit = T)

# summary of the network modules (colors represent module assignment)
net_df = data.frame(net$colors) # convert to a df
net_df = cbind(id = rownames(net_df), net_df) # change the index (node names) to a column
rownames(net_df) = 1:nrow(net_df)
colnames(net_df)[2] = 'louvain_label' # change column name

# change the file name below to wgcna_modules.csv during test
write.csv(net_df, './Data/pipeline/human/network_analysis/wgcna_modules.csv', row.names = F)