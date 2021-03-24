# Script to generate WCGNA network
library(WGCNA)
library("rjson")
args = commandArgs(trailingOnly = T)
config_file = args[1]
archive_path = args[2]

json_data = fromJSON(file = config_file)
expression = read.table(json_data[['inputs']][['normalized_counts']], header = TRUE)
rownames(expression) = expression$id
expression$id = NULL
expression_t = t(expression)
tom_output = file.path(archive_path, 'tom')

is_mouse = json_data[['parameters']][['skip_tom']] == FALSE || length(json_data[['parameters']][['skip_tom']]) == 0
if (is_mouse == TRUE) {saveTOMs = T} else {saveTOMs = F}
net = blockwiseModules(expression_t, power = 14, maxBlockSize = 30000,
                       TOMType = "unsigned", minModuleSize = 100,
                       reassignThreshold = 0, detectCutHeight = 0.99,
                       numericLabels = TRUE, pamRespectsDendro = FALSE,
                       saveTOMs = saveTOMs,
                       saveTOMFileBase = tom_output,
                       verbose = 3, deepSplit = T)

# load tom data
if (is_mouse) {
    tom_path = paste(tom_output, '-block.1.Rdata', sep = '')
    load(tom_path)
    tom_df = as.matrix(TOM)
    # add the gene IDs to the tom file
    colnames(tom_df) = colnames(expression_t)
    rownames(tom_df) = colnames(expression_t)
    # write tom file
    write.csv(tom_df, file = json_data[["outputs"]][['provided_networks']])
}
# summary of the network modules (colors represent module assignment)
net_df = data.frame(net$colors) # convert to a df
net_df = cbind(id = rownames(net_df), net_df) # change the index (node names) to a column
rownames(net_df) = 1:nrow(net_df)
colnames(net_df)[2] = 'louvain_label' # change column name

# change the file name below to wgcna_modules.csv during test
# write network modules 
write.csv(net_df, json_data[["outputs"]][["gene_to_module_mapping"]], row.names = F)
