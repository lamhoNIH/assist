# Script to generate WCGNA network
library(WGCNA)
library("rjson")
args = commandArgs(trailingOnly = T)
config_file = args[1]
archive_path = args[2]

json_data = fromJSON(file = config_file)
expression = read.table(file.path('./Data', json_data[['inputs']][['normalized_counts']]), header = TRUE)
rownames(expression) = expression$id
expression$id = NULL
expression_t = t(expression)
tom_output = file.path(archive_path, 'tom')

if (json_data[['parameters']][["skip_tom"]] == FALSE || length(json_data[['parameters']][['skip_tom']]) == 0) {
  saveTOMs = T} else {saveTOMs = F}

net = blockwiseModules(expression_t, power = 14, maxBlockSize = 30000,
                       TOMType = "unsigned", minModuleSize = 100,
                       reassignThreshold = 0, detectCutHeight = 0.99,
                       numericLabels = TRUE, pamRespectsDendro = FALSE,
                       saveTOMs = saveTOMs,
                       saveTOMFileBase = tom_output,
                       verbose = 3, deepSplit = T)

# load tom data
if (json_data[["parameters"]][["skip_tom"]] == FALSE || length(json_data[["parameters"]][['skip_tom']]) == 0) {
    tom_path = paste(tom_output, '-block.1.Rdata', sep = '')
    load(tom_path)
    tom_df = as.matrix(TOM)
    # add the gene IDs to the tom file
    colnames(tom_df) = colnames(expression_t)
    rownames(tom_df) = colnames(expression_t)
    # write tom file
    write.csv(tom_df, file = file.path(archive_path, 'tom.csv'))
}
# summary of the network modules (colors represent module assignment)
net_df = data.frame(net$colors) # convert to a df
net_df = cbind(id = rownames(net_df), net_df) # change the index (node names) to a column
rownames(net_df) = 1:nrow(net_df)
colnames(net_df)[2] = 'louvain_label' # change column name

# change the file name below to wgcna_modules.csv during test
# write network modules 
write.csv(net_df, file.path(archive_path, json_data[["outputs"]][["gene_to_module_mapping"]]), row.names = F)



# json_data <- fromJSON(file='./Data/pipeline/human/network_analysis/config.json')
#Load expression df with normalized count and the network ID file
# expression = read.table(file.path('./Data', json_data['normalized_counts']), header = TRUE)
# network_IDs = read.csv('./Data/pipeline/human/network_analysis/network_IDs.csv', row.names = 1)
# 
# expression = read.table('G:/Shared drives/NIAAA_ASSIST/Data/HDID_data/PFC_HDID_norm_exp.txt', header = TRUE)
# kapoor_expression = read.table('G:/Shared drives/NIAAA_ASSIST/Data/kapoor2019_batch.age.rin.sex.pm.alc.corrected.coga.inia.expression.txt', header = TRUE)
# ## Filter expression for network only expression
# # network_only_expression = expression[expression$id %in% network_IDs$X0,]
# 
# # Convert the id column to index and delete id column
# rownames(expression) = expression$probeID
# expression$probeID = NULL
# 
# ## automatic block-wise network
# # transpose the dataframe
# expression_t = t(expression)
# # network_only_expression_t = t(network_only_expression)
# # blockwiseModules() will generate Tom network along with the module detection by WCGNA
# net = blockwiseModules(expression_t, power = 14,
#                        TOMType = "unsigned", minModuleSize = 100,
#                        reassignThreshold = 0, detectCutHeight = 0.99,
#                        numericLabels = TRUE, pamRespectsDendro = FALSE,
#                        saveTOMs = F,
#                        #   saveTOMFileBase = "femaleMouseTOM",
#                        verbose = 3, deepSplit = T)
# 
# # summary of the network modules (colors represent module assignment)
# net_df = data.frame(net$colors) # convert to a df
# net_df = cbind(id = rownames(net_df), net_df) # change the index (node names) to a column
# rownames(net_df) = 1:nrow(net_df)
# colnames(net_df)[2] = 'louvain_label' # change column name
# 
# # change the file name below to wgcna_modules.csv during test
# # write.csv(net_df, './Data/pipeline/human/network_analysis/wgcna_modules.csv', row.names = F)
# write.csv(net_df, 'G:/Shared drives/NIAAA_ASSIST/Data/HDID_data/eda_derived/PFC_wgcna_modules.csv', row.names = F)
