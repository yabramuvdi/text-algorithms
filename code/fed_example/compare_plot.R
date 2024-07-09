library(rstudioapi)
library(dplyr)
library(ggplot2)
library(ggrepel)

####### setup

# Get the path of the active document
current_script_path <- getActiveDocumentContext()$path

# Set the working directory to the directory of the current script
setwd(dirname(current_script_path))

# Print the current working directory to verify
print(getwd())

# main paths
data_path <- "../../output/"
output_path <- "../../output/"

####### Data

df <- read.csv(paste0(data_path, "fed_compare.csv"))

# Define the mapping between clean_name and model_category
model_mapping <- data.frame(
  clean_name = c("Dictionary", "DistilBert", "Gemini 1.5", "GPT-3.5", "GPT-4o", "Logistic Regression", "Llama-3-8B",
                 "Claude-3.5", "Gemma-2-9B"),
  method_type = c("Basic", "Finetuned LLM", "Zero-shot LLM", "Zero-shot LLM", "Zero-shot LLM", "Basic", "Zero-shot LLM",
                  "Zero-shot LLM", "Zero-shot LLM"
                  )
)

df <- df %>%
  left_join(model_mapping, by = "clean_name")

####### Plot

custom_palette <- c("Basic" = "#1f77b4", "Finetuned LLM" = "#ff7f0e", "Zero-shot LLM" = "#2ca02c")

# Create the plot
ggplot(df, aes(x = num_params, y = error_rate, label = clean_name, color = method_type)) +
  geom_point(size = 3) +            # Points colored by 'dominated'
  geom_text_repel(size = 5, fontface = "bold") +            # Labels with legible font size
  scale_x_log10() +                                         # Log scale for x-axis
  scale_color_manual(values = custom_palette) +              # Apply the custom color palette
  geom_line(data = filter(df, dominated == "False"),        
            aes(x = num_params, y = error_rate, group = 1), 
            color = "black", linetype = "dashed") +
  labs(
    title = "FED data",
    x = "Number of Parameters (log scale)",
    y = "Error Rate",
    color = "Dominated"
  ) +
  theme_minimal(base_size = 15) +
  theme(legend.position = "none")  
