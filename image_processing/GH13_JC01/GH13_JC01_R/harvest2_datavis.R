## These script includes daata visualization for Harvest 2 from GH13JC01 Experiment ###

library(tidyverse)
library(dplyr)
library(readxl)
library(ggridges)
library(reshape2)
library(RColorBrewer)

raw_data<- read.csv("harvest2_data.csv") %>% select(1:8)

raw_data$tray_id = as.factor(raw_data$tray_id)
# Convert the raw data to a new data set where measuring variable names are included in a column.
# melt creates variable and values columns. Strings inside measure.vars taken as variable factor for variable.  

raw_reshape <- melt(raw_data, 
                      measure.vars = c("LFW_g","LDW_g", "LA_mm2"))


# Part 1:  Population Information / Data Distribution
## Use the reshaped data to facet using variable name.  
raw_reshape %>% 
  ggplot(aes(tray_id,value, fill = tray_id)) +
  geom_boxplot(outlier.shape = NA, width=0.5)+
  geom_jitter(width=0.15, alpha=0.5)+
  facet_wrap(~variable, scales = 'free') +
  scale_fill_brewer(palette = "Set1") + 
  theme(plot.title = element_text(size=16,hjust = 0.5, face="bold"),
      axis.text = element_text(size = 7,color="black"),
      axis.title.x = element_text(size=12,face="bold"),
      axis.title = element_text(size = 12, face="bold"),
      legend.position = "right",
      legend.background = element_rect(fill="white",
                                       size=1, linetype="solid", 
                                       colour ="black"),
      legend.text = element_text(size = 12),
      legend.title = element_text(size = 13,
                                  face = "bold"),
      strip.background = element_rect(color="black", fill="black", 
                                      size=1.5, linetype="solid"),
      strip.text.x = element_text(
        size = 12, color = "white", face = "bold.italic"))+
  labs(title="Harvest 2: Manual Measurements ",
       subtitle = "Date: 2023-09-02",
       fill="Tray",
       caption=NULL,
       x=NULL,
       y="g /g/ mm2")
    
                        

  