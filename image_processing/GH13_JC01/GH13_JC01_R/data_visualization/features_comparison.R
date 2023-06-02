# Comparison extracted geometrical traits and response variables
# Step 1: Show correlations between 

library(tidyverse)
library(dplyr)
library(readxl)
library(ggpubr)
library(ggridges)
library(reshape2)
library(lubridate)
library(RColorBrewer)
library(corrplot)
library(GGally)
library(caret)

# Create 4 data frames: 
# 1) Predictors and all response variables (raw_data), 
# 2) Just response variables (LFW, LA)
# 3) Image derived predictors and all response variables 
# 4) Image derived predictors and LFW
# 5) Image derived predictors and LDW
# 6) Image derived predictors and LA

raw_data<- read.csv("harvest_complete.csv")
raw_data$treatment <-as.factor(with(raw_data, ifelse(tray_id == '1', '280 ppm',
                                                     ifelse(tray_id == 2, '160 ppm', '80 ppm'))))
#raw_data$date <- dmy(raw_data$date)

img_predictors <- raw_data %>% select(1,3,12:25)
manual_predictors <- raw_data %>% select(1,3,7,9:12,25)
img_LFW <- raw_data %>% select(1,7,12:25)

img_BIC<- img_LFW %>% select("LFW_g" ,"plant_area", "plant_convex_hull_vertices", "plant_perimeter","plant_solidity","height_mm","plant_ellipse_minor_axis")
# Here I'm assuming that height was derived from image analysis.

# Provide a graph of the behavior of Leaf Fresh Weight (LFW) across time

img_LFW %>% ggplot(aes(x = date, y = LFW_g, group = date)) + 
  #geom_point(alpha = 0.3) + 
  #facet_wrap(~ variable, scales = 'free') + 
  geom_point(color = "blue", alpha = .3) +
  geom_boxplot()+
  #stat_summary(fun = "mean", geom = "point", color = "red", size = 1) +
  #stat_summary(fun = "mean", geom = "line", aes(group = 1), color = "red") +
  labs(x = "Date", y = "Leaf Fresh Weight (g)") +
  scale_x_date(date_breaks = "7 day") +
  theme_bw() +
  scale_color_brewer(palette = "Set2")

# Reshape the data in such a way that all geometrical features will be in a single column.
# By doing this, I'm able to use facet wrap with geometrical features, to have a regression plot for each feature vs response variable. 

raw_reshape <- melt(raw_data, 
                    measure.vars = c("plant_area", "plant_convex_hull_vertices", "plant_perimeter","plant_solidity","height_mm","plant_ellipse_minor_axis"))
# Correlation between image derived features and Leaf Fresh Weight.  
## Here I'm fitting  a logarithmic regression. 

raw_reshape %>% ggplot(aes(x = LFW_g, y = value)) + 
  geom_point(alpha = 0.3) + 
  facet_wrap(~ variable, scales = 'free_y') + 
  geom_smooth(method = lm, formula = y ~ x, color = "red",fill = "black") +
  stat_cor(method = "pearson", aes(label = ..r.label..),label.x.npc = c("left"),label.y.npc = c("top"))+
  labs(x = "Leaf Fresh Weight (g)", 
       y = NULL, 
       title ="Image-Derived Features",
       color="Geometrical Features",
       caption=NULL) +
  theme_bw() +
  theme(plot.title = element_text(size=16,hjust = 0.5, face="bold"),
        axis.text = element_text(size = 7,color="black"),
        axis.title.x = element_text(size=12,face="bold"),
        axis.title = element_text(size = 12, face="bold"),
        legend.position = "left",
        legend.background = element_rect(fill="white",
                                         size=0.5, linetype="solid", 
                                         colour ="black"),
        legend.text = element_text(size = 10),
        legend.title = element_text(size = 9,
                                    face = "bold"),
        strip.background = element_rect(color="black", fill="gray", 
                                        size=1, linetype="solid"),
        strip.text.x = element_text(
          size = 9, color = "black", face = "bold"))+
  scale_color_brewer(palette = "Set1")

ggsave("LFW_correlations.tiff",width = 7,height = 5,units = c("in"))








# Plot a correlation between all the predictors and response variable. 
## From these graphs we can see that data distribution of most features are not normally distributed. 
LFW_cor<- raw_data %>% select("LFW_g","plant_area", "plant_convex_hull_area","plant_solidity", "plant_perimeter","plant_width","plant_height","plant_longest_path")

ggpairs(img_BIC)

 



