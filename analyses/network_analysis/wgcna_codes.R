# Script to generate WCGNA network

library(WGCNA)

## automatic block-wise network
# load expression data
expression = read.csv('./Data/eda_derived/network_only_expression.csv', 
                      header = T, row.names = 1)
# transpose the dataframe
expression_t = t(expression)

# blockwiseModules() will generate Tom network along with the module detection by WCGNA
net = blockwiseModules(expression_t, power = 14,
                       TOMType = "unsigned", minModuleSize = 100,
                       reassignThreshold = 0, detectCutHeight = 0.99,
                       numericLabels = TRUE, pamRespectsDendro = FALSE,
                       saveTOMs = F,
                       #   saveTOMFileBase = "femaleMouseTOM",
                       verbose = 3, deepSplit = T)

# summary of the network modules (colors represent module assignment)
# table(net$colors)

net_df = data.frame(net$colors) # convert to a df
net_df = cbind(id = rownames(net_df), net_df) # change the index (node names) to a column
rownames(net_df) = 1:nrow(net_df)
colnames(net_df)[2] = 'louvain_label' # change column name

# change the file name below to wgcna_modules.csv during test
write.csv(net_df, './Data/eda_derived/wgcna_modules_test.csv', row.names = F)