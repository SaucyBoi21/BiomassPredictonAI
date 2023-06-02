library(tidyverse)
library(dplyr)
library(readxl)
library(ggridges)
library(reshape2)
library(RColorBrewer)

tray1_data <- read_csv("tray1_database.csv")

tray1_data$Date <- as.Date(tray1_data$Date)


tray1_bydate<-tray1_data  %>% group_by(Date,plant_id) %>%
  summarise(mean_area = mean(plant_area),
            plants =  n())


tray1_bydate %>% filter(plant_id %in% c("plant0", "plant2"))%>%
  ggplot( aes(x=Date, y=mean_area,color =  plant_id)) +
  geom_line() + 
  geom_point()+
  xlab("") +
  theme_bw() +
  theme(axis.text.x=element_text(angle=60, hjust=1))


## Reshape dataframe to visualize one box plot per variable: 
tray1_reshape <- melt(tray1_data, 
                      measure.vars = c("plant_perimeter","plant_width","plant_height","plant_solidity","plant_longest_path"))


# Traits distribution for all tray
tray1_reshape %>% filter(Date == "2023-02-01") %>%
  ggplot(aes(value,variable,fill=variable))+
  geom_density_ridges()+
  theme_bw()+
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
  labs(title="Geometrical Traits per Tray",
       subtitle = "Average of 55 plants",
       fill="Geometrical Traits",
       caption=NULL,
       x="Pixels",
       y=NULL)+
  scale_fill_brewer(palette="Set1")

# Plant area for a single plant
tray1_data %>% select(plant_area,plant_id,Date) %>% 
  filter(plant_id %in% c("plant10", "plant1", "plant30", "plant40"),
  Date == "2023-02-02") %>% 
  ggplot(aes(x=plant_id, y=plant_area, fill = plant_id)) +
  geom_col() + 
  geom_point()+
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
  labs(title="Area in Pixels per Plant",
       subtitle = "Date: 2023-02-02",
       fill="Plant ID",
       caption=NULL,
       x=NULL,
       y="Shoot Area (pixels)")+
  scale_fill_brewer(palette="Set1")

  