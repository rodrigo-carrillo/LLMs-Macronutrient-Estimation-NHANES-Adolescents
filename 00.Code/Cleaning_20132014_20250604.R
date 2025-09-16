
library(haven)
library(dplyr)
rm(list = ls())



df1 <- read_xpt('C:/Users/rmcarri/OneDrive - Emory University/NHANES/00.dietaryindex/NHANES 2013-2014/DR1IFF_H.xpt')
df2 <- read_xpt('C:/Users/rmcarri/OneDrive - Emory University/NHANES/00.dietaryindex/NHANES 2013-2014/DR2IFF_H.xpt')
dfc <- read_xpt('C:/Users/rmcarri/OneDrive - Emory University/NHANES/00.dietaryindex/NHANES 2013-2014/DRXFCD_H.xpt')



df1$DR1IFDCD <- as.character(df1$DR1IFDCD)
df2$DR1IFDCD <- as.character(df2$DR2IFDCD)
dfc$DR1IFDCD <- as.character(dfc$DRXFDCD)
dfc$DR1IFDCD <- NULL



df1b <- merge(
  df1,
  dfc,
  by.x = 'DR1IFDCD',
  by.y = 'DRXFDCD',
  all.x = TRUE
)
df2b <- merge(
  df2,
  dfc,
  by.x = 'DR2IFDCD',
  by.y = 'DRXFDCD',
  all.x = TRUE
)



df1b <- df1b %>%
  mutate(day = 1) %>%                      # To identify the day.
  filter(DR1DRSTZ == 1,                    # Only high-quality recalls.
         DRABF == 2) %>%                   # Only not breastfeeding.
  select(SEQN, day, DR1CCMNM, DR1CCMTX, 
         DR1_020, DR1_030Z, DR1FS, DR1_040Z,
         DR1IFDCD, DR1IGRMS, DR1IKCAL, DR1IPROT, 
         DR1ICARB, DR1ISUGR, DR1IFIBE, DR1ITFAT, 
         DRXFCLD, DRXFCSD)
df2b <- df2b %>%
  mutate(day = 2) %>%
  filter(DR2DRSTZ == 1,
         DRABF == 2) %>%
  select(SEQN, day, DR2CCMNM, DR2CCMTX, 
         DR2_020, DR2_030Z, DR2FS, DR2_040Z,
         DR2IFDCD, DR2IGRMS, DR2IKCAL, DR2IPROT, 
         DR2ICARB, DR2ISUGR, DR2IFIBE, DR2ITFAT, 
         DRXFCLD, DRXFCSD)



colnames(df1b) <- gsub("^DR[12]", "DRx", colnames(df1b))
colnames(df2b) <- gsub("^DR[12]", "DRx", colnames(df2b))



df <- rbind(
  df1b, df2b
)
nrow(df1b) + nrow(df2b) == nrow(df)



df_collapsed <- df %>%
  group_by(SEQN, day) %>%
  summarise(
    diet = paste0(DRXFCSD, " (", DRxIGRMS, ")", collapse = "; "),
    DRxIKCAL = sum(DRxIKCAL, na.rm = TRUE),
    DRxIPROT = sum(DRxIPROT, na.rm = TRUE),
    DRxICARB = sum(DRxICARB, na.rm = TRUE),
    DRxISUGR = sum(DRxISUGR, na.rm = TRUE),
    DRxIFIBE = sum(DRxIFIBE, na.rm = TRUE),
    DRxITFAT = sum(DRxITFAT, na.rm = TRUE)
  ) %>%
  ungroup()
length(unique(df$SEQN))
length(unique(df_collapsed$SEQN))
all(unique(df_collapsed$SEQN) %in% unique(df$SEQN))
all(unique(df$SEQN) %in% unique(df_collapsed$SEQN))

################################################################################





### CLINICAL DATA & MERGE ######################################################

df_clinical <- read.csv('C:/Users/rmcarri/OneDrive - Emory University/NHANES/00.Data/_extracted/NHANES_2013_2014__2024-12-17.csv') %>%
  select(
    id, id_study, age, sex, race, pir, edu, is_preg,
    self_db, self_ht,
    smoker
  )



df_collapsed <- merge(df_collapsed,
                      df_clinical,
                      by.x = c('SEQN'),
                      by.y = c('id'),
                      all.x = TRUE)

################################################################################





### SAVE FINAL DATASET #########################################################

write.csv(df_collapsed,
          paste0('C:/Users/rmcarri/OneDrive - Emory University/NHANES/Macronutrients LLMs/02.Datasets/df_20132014_', Sys.Date(), '.csv'),
          row.names = FALSE)

################################################################################


