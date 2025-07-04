# ---------------------------------------------------------------------------- #
#                         Michael Duarte Gonçalves                             #
#                             Pre-doc RA, UC3M                                 #
#                                July 2025:                                    #
#                       Numerical Methods Exercise                             #
#                             With True Data                                   #
# ---------------------------------------------------------------------------- #

# Packages ----------------------------------------------------------------

# Load and install the packages that we'll be using
if (!require("pacman")) install.packages("pacman")
pacman::p_load(
  ggplot2, dplyr, furrr, 
  tidyr, ggrepel, glue, here,
  mapSpain, purrr, tibble, 
  readr, readxl, writexl, openxlsx,
  stringr, fs, janitor, stats, 
  rootSolve, scales, patchwork, sf
)

# Remove any objects in memory
rm(list = ls())

# Set Working Directory ---------------------------------------------------

# Set working directory to "NumericalMethods_NF_GL" relative to the current script's location (code folder)

working_dir <- here()


# ---- 1. Base output directory paths

# Define Dynamic Paths for Input and Output Files
  # Inputs
data_raw <- file.path(working_dir, "data")

out_tables <- file.path(working_dir, "out", "tables")
out_figures <- file.path(working_dir, "out", "figures")

counterparty_risk_dir_fig <- file.path(out_figures, "counterparty_risk")
no_cpr_dir_fig <- file.path(out_figures, "no_counterparty_risk")

counterparty_risk_tab_dir <- file.path(out_tables, "counterparty_risk")
no_cpr_tab_dir <- file.path(out_tables, "no_counterparty_risk")

# ---- 2. Create top-level output folders if missing
dir_list <- list(
  counterparty_risk_dir_fig,
  counterparty_risk_tab_dir,
  no_cpr_dir_fig,
  no_cpr_tab_dir
)

for (dir_path in dir_list) {
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE)
    cat("Created subfolder:", dir_path, "\n")
  } else {
    cat("Subfolder already exists:", dir_path, "\n")
  }
}

# ---- 3. Define subproject folder
subfolder_name <- "wind_solar_proj_2022"

new_dirs <- list(
  counterparty_risk_fig = file.path(counterparty_risk_dir_fig, subfolder_name),
  counterparty_risk_tab = file.path(counterparty_risk_tab_dir, subfolder_name),
  no_cpr_fig = file.path(no_cpr_dir_fig, subfolder_name),
  no_cpr_tab = file.path(no_cpr_tab_dir, subfolder_name)
)

# Create wind_solar_proj_2022 folders
for (dir_path in new_dirs) {
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE)
    cat("Created directory:", dir_path, "\n")
  } else {
    cat("Directory already exists:", dir_path, "\n")
  }
}


new_dirs_cpr <- list(
  counterparty_risk_tab = file.path(counterparty_risk_tab_dir, subfolder_name)
)

# ---- 4. Create sub-subfolders inside wind_solar_proj_2022
sub_subfolders <- c("Baseline", "Public_Subsidies", 
                    "Public_Guarantees", "Regulator_Backed_Contracts")

for (dir_path in new_dirs_cpr) {
  for (subfolder in sub_subfolders) {
    full_path <- file.path(dir_path, subfolder)
    if (!dir.exists(full_path)) {
      dir.create(full_path, recursive = TRUE)
      cat("Created sub-subfolder:", full_path, "\n")
    } else {
      cat("Sub-subfolder already exists:", full_path, "\n")
    }
  }
}

baseline_path <- file.path(counterparty_risk_tab_dir, "wind_solar_proj_2022", "Baseline")
with_T_path <- file.path(counterparty_risk_tab_dir, "wind_solar_proj_2022", "Public_Subsidies")
with_public_guarantees_path <- file.path(counterparty_risk_tab_dir, "wind_solar_proj_2022", "Public_Guarantees")
with_rbc_path <- file.path(counterparty_risk_tab_dir, "wind_solar_proj_2022", "Regulator_Backed_Contracts")


# ---- 5. Create dynamic theme-based sub-subfolder for figures
# Define subproject and themed subdirectory name

# We create a function that allows us to select the color that we want.
# Define base color palettes (dark-mid-light)
# This is useful for further plotting in either green or blue.

blue_palette_base <- c("#08306B", "#2171B5", "#9ECAE1")   # blue palette
green_palette_base <- c("#00441B", "#238B45", "#A1D99B")  # green palette


# Choose theme: "Blue" (paper) or "Green" (presentation)
selected_color_theme <- "Green"  # <-- Just change this line!

subfolder_name <- "wind_solar_proj_2022"
theme_subdir_name <- switch(
  selected_color_theme,
  "Blue" = "paper_version",
  "Green" = "presentation_version",
  stop("Unknown color theme.")
)

# Full base figure path, depending on theme
base_fig_dir <- file.path(counterparty_risk_dir_fig, subfolder_name, theme_subdir_name)

# Create main theme folder if needed
if (!dir.exists(base_fig_dir)) {
  dir.create(base_fig_dir, recursive = TRUE)
  cat("Created themed figure base dir:", base_fig_dir, "\n")
} else {
  cat("Themed figure base dir already exists:", base_fig_dir, "\n")
}

# Define scenario-level subdirectories
baseline_fig_path <- file.path(base_fig_dir, "Baseline")
with_T_fig_path <- file.path(base_fig_dir, "Public_Subsidies")
with_pg_fig_path <- file.path(base_fig_dir, "Public_Guarantees")
with_rbc_fig_path <- file.path(base_fig_dir, "Regulator_Backed_Contracts")

# Create all the sub-subfolders inside the theme folder
sub_fig_paths <- list(baseline_fig_path, with_T_fig_path, with_pg_fig_path, with_rbc_fig_path)

for (dir_path in sub_fig_paths) {
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE)
    cat("Created sub-subfolder:", dir_path, "\n")
  } else {
    cat("Sub-subfolder already exists:", dir_path, "\n")
  }
}

# ---- 6. Create theme-based directory for NO-CPR figures (no sub-subfolders)

no_cpr_base_fig_dir <- file.path(no_cpr_dir_fig, subfolder_name, theme_subdir_name)

if (!dir.exists(no_cpr_base_fig_dir)) {
  dir.create(no_cpr_base_fig_dir, recursive = TRUE)
  cat("Created themed no-CPR figure base dir:", no_cpr_base_fig_dir, "\n")
} else {
  cat("Themed no-CPR figure base dir already exists:", no_cpr_base_fig_dir, "\n")
}

# ---- 7. Define filename suffix based on theme. This is just for two figures:

  # i. solar_wind_avg_cost_cumulative_capacity (Section 2)
  # ii. spain_map_wind_solar_proj_2022 (Section 9)

file_suffix <- switch(
  selected_color_theme,
  "Blue" = "_paper",
  "Green" = "_presentation",
  stop("Unknown color theme. Choose 'Green' or 'Blue'.")
)

# Set wd for data_raw

setwd(data_raw)

# Section 1: Create Dataset -----------------------------------------------


# We define now some key parameters for dataset creation

fx <- 0.94895             #USD/EU   # Exchange Rate
c_inv_solar_usd <- 778    #USD/kW   # Investment Cost(solar)
c_inv_wind_usd <- 1159    #USD/kW   # Investment Cost (wind)
c_om_solar_usd <- 7.36    #USD/kW   # Operation & Maintanance Cost (s)
c_om_wind_usd <- 29.9     #USD/kW   # Operation & Maintanance Cost (w)

c_inv_s <- c_inv_solar_usd / fx # EUR/kW # Investment cost (solar)
c_inv_w <- c_inv_wind_usd  / fx # EUR/kW # Investment cost (wind)
c_om_s <- c_om_solar_usd   / fx # EUR/kW # Operation & Maintenance Cost (solar)
c_om_w <- c_om_wind_usd    / fx # EUR/kW # Operation & Maintenance Cost (wind)

omega <- 0.01             # baseline omega
epsilon <- 0              # cost shock for technology t. Now = 0.
life <- 25                # lifetime of a plant
v_mw <- 400               # Euro/MWh. Consumer valuation (v from buyer profit function)

# ---------------------------------------------------------------------------- #

# Load key datasets
# Important: please have 
#   i. "Wind_Solar_projects_Spain_2022 - merged.csv", and
#  ii. "Wind_Solar_projects_with_coordinates_Spain_2022.xlsx"
#  in the data_raw folder, please.

file_name <- "Wind_Solar_projects_Spain_2022 - merged.csv"

file_path <- fs::path(data_raw, file_name)

data_solar_wind_proj_2022 <- read_csv(file_path)

# Some plants are really big, so we decide to split them, in order to have better results.
# 
#  Max capacity per plant
max_capacity <- 50

# This function now accepts a grouped tibble (all rows with the same projectname)
split_project <- function(df_group) {
  total_capacity <- sum(df_group$capacity)

  if (total_capacity <= max_capacity) {
    return(df_group)
  }

  n_splits <- ceiling(total_capacity / max_capacity)
  cap_split <- rep(floor(total_capacity / n_splits * 1000) / 1000, n_splits)
  cap_split[length(cap_split)] <- round(total_capacity - sum(cap_split[-length(cap_split)]), 3)

  tibble(
    projectname = str_c(unique(df_group$projectname), " ", seq_len(n_splits)),
    capacity = cap_split,
    avgcapacityfactor = rep(df_group$avgcapacityfactor[1], n_splits),
    type = rep(df_group$type[1], n_splits)
  )
}

# Apply: first group by projectname
data_solar_wind_proj_2022 <- data_solar_wind_proj_2022 |>
  group_by(projectname) |>
  group_split() |>
  map_dfr(split_project) |>
  ungroup()



# Apply to dataset with pmap_dfr


wind_solar_proj_2022 <- data_solar_wind_proj_2022 |> 
  mutate(
    hours = (avgcapacityfactor * 8760),       # Total hours worked in a year
    power_kw = capacity * 1000,               # Capacity (kW)
    q_i_kwh = hours * life * power_kw,        # Total production (kWh)
    q_i_mwh = hours * life * capacity,        # Total production (MWh)
    v_q_i_mwh = v_mw * q_i_mwh                # (Euro/MWh)*MWh = Euro
) |> 
  mutate(
    c_inv = case_when(
      type == "Solar" ~ c_inv_s,
      type == "Wind"  ~ c_inv_w
    ),
    c_om = case_when(
      type == "Solar" ~ c_om_s,
      type == "Wind"  ~ c_om_w
    ),
    total_cost = (c_inv + c_om * life + epsilon) * (power_kw + (1000 - power_kw) * omega),
    avg_cost_euro_kwh = (total_cost / q_i_kwh),
    avg_cost_euro_mwh = (total_cost / q_i_mwh)
  ) |> 
  arrange(projectname, capacity)

# ---------------------------------------------------------------------------- #

## Key Parameters ---------------------------------------------------------

# Key Parameters for Subsequent Sections

x <- 60                 # parameter x (scaling factor). 

alpha <- 4              # alpha = 4 (beta distribution param.)

beta <- 2               # beta = 2 (beta distribution param.)

# p \in (0,1) and follows a Beta distribution with params.:

# i.  alpha, beta, 
# ii. expected_p and var_p 

expected_p <- (alpha / (alpha + beta)) # E(p) from a beta distribution

var_p <- (alpha*beta) / ((alpha+beta)^2*(alpha+beta+1)) # Var(p) from a beta distribution

threshold_price <- x * expected_p # threshold price

# gamma_values  # share of opportunistic buyers
gamma_values <- seq(0, 0.5, by = 0.025)

# Compute an arbitrarily r_0 
r_0 <- 1.334653e-07

# per-unit cost of social funds.
lambda <- 0.3                   

# lambda enters in:
# Section 5: Public Guarantees
# Section 6: Public Subsidies
# Section 7: Public Guarantees/Subsidies Comparison

# For simplicity's sake, we only preserve for the subsequent saved tables 
# the following gammas:

# 0, 
# 0.1,
# 0.2,
# 0.3, 
# 0.4,
# 0.5

# For Section 3 (No-CPR) only:
theta_no_cpr <- 3500    # theta --> contract demand. Arbitrarily chosen at 3500.


# For Section 4 to Section 7 (various CPR scenarios):
theta_values <- c(2500, 3500, 4500)  # Theta values (aka contract demand) to loop over

# For Section 6: for this section, we have public subsidies T. To avoid the
# figures to be too crowded, we select a few gammas and a theta

selected_gammas <- seq(from = 0, to = 0.5, by = 0.1) 

selected_theta <- 3500

# Section 8:
# In the case of regulator-backed contracts, we have:

# i.   the total contract demand (sum of private + RBC)
# ii.  the RBC demand 

total_contract_demand <- 5000

private_demand <- 2500

rbc_demand <- 2500

theta_comparison_rbc <- 2500


# For all plots, we want to use the same base of fonts for each:

base_s <- 25

## Color Palette ----------------------------------------------------------


# Assign base palette based on selected theme
base_palette <- switch(
  selected_color_theme,
  "Green" = green_palette_base,
  "Blue" = blue_palette_base,
  stop("Unknown color theme. Choose 'Green' or 'Blue', please.")
)

theme_palette_gamma <- gradient_n_pal(base_palette)(seq(0, 1, length.out = length(gamma_values)))
theme_palette_theta <- gradient_n_pal(base_palette)(seq(0, 1, length.out = length(theta_values)))
theme_palette_welfare <- gradient_n_pal(base_palette)(seq(0, 1, length.out = 3))
theme_palette_avg_cost_graphs <- base_palette[2]
theme_palette_map <- c("Wind" = base_palette[1], "Solar" = base_palette[3])


# Section 2: Compute Cumulative G(.) Functions ----------------------------

# Note that G functions are not CDFs. 
# These are giving nominal production & capacity levels.
  
  # Create G_k: in terms of MW
  # Create G_Q: in terms of MWh

# Create G_k
G_k <- wind_solar_proj_2022 |> 
  arrange(avg_cost_euro_mwh) |> 
  group_by(avg_cost_euro_mwh) |> 
  summarise(cumulative_capacity = sum(capacity)) |> 
  mutate(cumulative_capacity = cumsum(cumulative_capacity)) |> 
  ungroup()

# Create G_Q
G_Q <- wind_solar_proj_2022  |> 
  arrange(avg_cost_euro_mwh) |> 
  group_by(avg_cost_euro_mwh) |> 
  summarise(cumulative_production = sum(q_i_mwh)) |> 
  mutate(cumulative_production = cumsum(cumulative_production)) |> 
  ungroup()

# We display the cumulative total production by the plant type. 

solar_proj_2022 <- wind_solar_proj_2022  |> 
  filter(type == "Solar") |> 
  arrange(avg_cost_euro_mwh)

G_k_solar <- data.frame(avg_cost_euro_mwh = numeric(), 
                        cumulative_capacity = numeric())

for (cost in unique(solar_proj_2022$avg_cost_euro_mwh)) {
  cumulative_capacity <- sum(solar_proj_2022$capacity[solar_proj_2022$avg_cost_euro_mwh 
                                                     <= cost])
  G_k_solar <- rbind(G_k_solar, 
                     data.frame(avg_cost_euro_mwh = cost, 
                                cumulative_capacity = cumulative_capacity))
}

wind_proj_2022 <- wind_solar_proj_2022  |> 
  filter(type == "Wind") |> 
  arrange(avg_cost_euro_mwh)


G_k_wind <- data.frame(avg_cost_euro_mwh = numeric(), 
                       cumulative_capacity = numeric())

for (cost in unique(wind_proj_2022$avg_cost_euro_mwh)) {
  cumulative_capacity <- sum(wind_proj_2022$capacity[wind_proj_2022$avg_cost_euro_mwh 
                                                         <= cost])
  G_k_wind <- rbind(G_k_wind, 
                    data.frame(avg_cost_euro_mwh = cost,
                               cumulative_capacity = cumulative_capacity))
}

G_k_wind$type <- 'Wind'

G_k_solar$type <- 'Solar'

G_k_solar_and_wind <- rbind(G_k_wind, G_k_solar)

G_k_all <- wind_solar_proj_2022 |> 
  arrange(avg_cost_euro_mwh) |> 
  group_by(avg_cost_euro_mwh) |> 
  summarise(capacity = sum(capacity, na.rm = TRUE)) |> 
  ungroup() |> 
  mutate(cumulative_capacity = cumsum(capacity))


avg_cost_vs_cum_cap <- ggplot(G_k_all, 
                              aes(x = cumulative_capacity,
                                  y = avg_cost_euro_mwh)) +
  geom_point(size = 8,
             shape = 17,
             alpha = 0.5,
             color = theme_palette_avg_cost_graphs) +  # Single color for all points #045A8D (blue) / #003d17 (green)
  geom_step(alpha = 0.3,
            size = 0.5,
            linetype = "solid",
            color = theme_palette_avg_cost_graphs) +   # Same color for the line #045A8D (blue) / #003d17 (green)
  labs(
    x = "Cumulative Capacity (MW)",
    y = "Avg. Cost (€/MWh)"
  ) +
  theme_minimal(base_size = base_s) 

avg_cost_vs_cum_cap

plot_filename <- paste0("solar_wind_avg_cost_cumulative_capacity", file_suffix, ".pdf")
plot_path_cpr <- file.path(out_figures, plot_filename)

# Save the plot
ggsave(
  filename = plot_path_cpr,
  plot = avg_cost_vs_cum_cap,
  width = 16,
  height = 9,
  dpi = 300
)


#  ---------------------------------------------------------------------- #

# Section 3: Modelling Markets w/o CPR ------------------------------------

# INSTRUCTIONS! --------------------------------------------------------- #
# We will now modeling markets without CPR

  # Spot market profits
    
    # Sellers get the following utilities:

      #  i.  \Pi_S^0(c) = q_i \times x \times E(p) - C(k_i) - r_i,
      #  ii. r_i = r_0 Var(pxq_i) = r_0(xq_i)^2 Var(p)

      #  iii. Be careful! r_0 should be sufficiently low, in order to 
      #  make sure that some plants find it optimal to trade in spot m.
      #  for at least some plants, we should have \Pi_S^0(c) > 0.


#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Functions

# Define some functions. Here, we want to make sure that at least 10% of the sample
# make profits in the spot market. For that, we will use some numerical 
# methods to find this minimal threshold
# N.B.: acronyms "sp" means "spot market" and "no_cpr" below means "no counterparty risk"

# Define function to check profit percentage for a given r_0
check_profit_proportion_sp_no_cpr <- function(r_0, df) {
  df <- df |> 
    mutate(r = r_0 * (x * q_i_mwh)^2 * var_p,  # Calculate r_i
           profits_sp_no_cpr = q_i_mwh * x * expected_p - total_cost - r)
  
  # Calculate the percentage of projects with positive profits
  mean(df$profits_sp_no_cpr > 0)
}

#  ---------------------------------------------------------------------- #
# ----------------------------------------------------------------------- #

# Function: Find the smallest r_0 that ensures at least 1% of projects are profitable
find_optimal_r0 <- function(df, lower = 1e-12, upper = 1e-3, target = 0.01, tol = 1e-12) {
  # Define function to find zero crossing
  f <- function(r_0) check_profit_proportion_sp_no_cpr(r_0, df) - target
  
  # Use uniroot instead of optimize
  result <- tryCatch(
    uniroot(f, interval = c(lower, upper), tol = tol)$root,
    error = function(e) return(NA)  # Return NA if no solution is found
  )
  
  return(result)
}

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Spot Market Profits

# Based on the previous created function, one could choose approximately the 
# percentage of the sample that we want to have positive profits
# by changing above the target value that we want.
# Example: now, the target is defined to 0.01, meaning that we want 1%
# of the sample to have positive profits.

# This is the aim of the following command. Compute the optimal r_0 given
# a target:

# NOTE: IF YOU WANT TO USE THIS r_0, please delete the one defined above
# in "Key Parameters" subsection of Section 1.

# r_0 <- find_optimal_r0(wind_solar_proj_2022)

wind_solar_proj_2022 <- wind_solar_proj_2022 |> 
  mutate(r_0 = r_0,
         r = r_0 * (x*q_i_mwh)^2 * var_p,
         profits_sp_no_cpr = q_i_mwh * x * expected_p - total_cost - r
  )


# Check final profit percentage
profit_percentage <- mean(wind_solar_proj_2022$profits_sp_no_cpr > 0)*100
cat("Final percentage of profitable projects:", profit_percentage, "%\n")

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Contracts w/o counterparty risk

wind_solar_proj_2022_no_cpr <- wind_solar_proj_2022 |> 
  mutate(
    f_c    = total_cost / (q_i_mwh * x),  # Break-even contract price
    f_spot = (x * expected_p - r_0 * var_p * q_i_mwh * x^2) / x,  # Spot market constraint
    f_max = pmax(f_c, f_spot),  
    xf_c = x * f_c,
    xf_spot = f_spot * x,
    xf_max  = pmax(xf_c, xf_spot) # Take the max of the two
  )

# Now, we define q_0 as the sum of the total production for plants that 
# have positive profits

# Identify the chosen xf for positive profits
wind_solar_proj_2022_no_cpr <- wind_solar_proj_2022_no_cpr |> 
  mutate(
    chosen_xf = case_when(
      profits_sp_no_cpr > 0 & xf_max == xf_c ~ "xf_c",
      profits_sp_no_cpr > 0 & xf_max == xf_spot ~ "xf_spot",
      TRUE ~ "xf_c"
    )
  ) |> 
  mutate(x = x,
         expected_p = expected_p,
         var_p = var_p) |> 
  select(projectname, capacity, x, expected_p, var_p, everything())

# View results for positive profits (Check if xf_spot is always chosen).
# It is the case

wind_solar_proj_2022_no_cpr |> 
  filter(profits_sp_no_cpr > 0) |> 
  select(profits_sp_no_cpr, xf_c, xf_spot, chosen_xf) |> 
  head()

q_0 <- wind_solar_proj_2022_no_cpr |> 
  filter(profits_sp_no_cpr > 0) |> 
  summarise(q_0 = sum(q_i_mwh)) |> 
  pull()


# Create a dataset with the contract supply curve
# Here, we will add some important columns:

  # i. q_0 --> firms that have positive spot market profits. q_0 represents
    # their cumulative production.

  # ii. G_expected_p_x_value: the sum of output for prices below x·E(p)

contract_supply_nocpr <- wind_solar_proj_2022_no_cpr |> 
  group_by(xf_max) |> 
  summarise(
    total_capacity = sum(capacity),      # Sum capacity for this contract price
    total_production = sum(q_i_mwh)     # Sum production for this contract price
  ) |> 
  arrange(xf_max) |>                     # Ensure ordering by price
  mutate(
    cumulative_capacity = cumsum(total_capacity),     # Cumulative sum of capacity
    cumulative_production = cumsum(total_production), # Cumulative sum of total production
    q_0 = q_0,
    G_expected_p_x = cumsum(if_else(xf_max <= expected_p * x,
                                    total_capacity, 0)) # Sum production for prices below expected_p·x
  ) |> 
  mutate(
    G_expected_p_x = last(G_expected_p_x, order_by = xf_max)
  ) |> 
  ungroup()

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Add Contract Demand Variable
# We now define a contract demand that should be greater than q_0:

# We subsequently will create a dataset containing this parameter value.

contract_supply_nocpr <- contract_supply_nocpr |> 
  mutate(contract_demand = if_else(xf_max <= x * expected_p, theta_no_cpr, 0),  # Adds the same theta value to all rows
         )
# Define the path
excel_path_contract_s_d <- file.path(no_cpr_tab_dir, "wind_solar_proj_2022", "02_contract_supply_demand_nocpr.xlsx")

# Save dataset as an Excel file
write_xlsx(contract_supply_nocpr, path = excel_path_contract_s_d)

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Plot cumulative production vs. xf_contract
cumul_prod_xfmax <- ggplot() +
  # Contract Supply Curve (Step Function)
  geom_step(data = contract_supply_nocpr, aes(x = cumulative_production, y = xf_max), 
            color = theme_palette_avg_cost_graphs, size = 1) +
  # Labels and theme
  labs(x = "Cumulative Production (MWh)",
       y = expression("Contract Price (€/MWh)")) +
  theme_minimal(base_size = base_s) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

# Plot cumulative capacity vs. xf_contract
cumul_cap_xfmax <- ggplot() +
  # Contract Supply Curve (Step Function)
  geom_step(data = contract_supply_nocpr, aes(x = cumulative_capacity, y = xf_max), 
            color = theme_palette_avg_cost_graphs, size = 1) +
  # Labels and theme
  labs(x = "Cumulative Capacity (MW)",
       y = "") +
  theme_minimal(base_size = base_s) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

cumul_graphs <- cumul_prod_xfmax + cumul_cap_xfmax

cumul_graphs

cumul_supply_path <- file.path(no_cpr_base_fig_dir, "01_supply_no_cpr.pdf")

ggsave(cumul_supply_path, plot = cumul_graphs, width = 16, height = 9, dpi = 300)

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Equilibrium Prices/Quantities

# Create demand segment for plotting
demand_segments <- tibble(
  x_start = theta_no_cpr,
  x_end   = theta_no_cpr,
  y_start = min(contract_supply_nocpr$xf_max),
  y_end   = max(contract_supply_nocpr$xf_max)
)

# Generate the plot
supply_demand_plot <- ggplot() +
  geom_step(data = contract_supply_nocpr, aes(x = cumulative_capacity, y = xf_max), 
            color = theme_palette_avg_cost_graphs, size = 1) +
  geom_segment(data = demand_segments, aes(x = x_start, xend = x_end, 
                                           y = y_start, yend = y_end),
               color = "black", size = 1, linetype = "solid") +
  labs(x = "Cumulative Capacity (MW)",
       y = expression("Contract Price (€/MWh)")) +
  theme_minimal(base_size = base_s) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

supply_demand_plot

# Save the plot
plot_path <- file.path(no_cpr_base_fig_dir, "02_supply_demand_no_cpr.pdf")
ggsave(plot_path, plot = supply_demand_plot, width = 16, height = 9, dpi = 300)


equilibrium_nocpr <- contract_supply_nocpr |> 
  filter(cumulative_capacity >= theta_no_cpr) |> 
  slice(1) |> 
  mutate(equilibrium_quantity = theta_no_cpr) |>  # Store theta as equilibrium quantity
  select(xf_max, equilibrium_quantity) |> 
  rename(equilibrium_price = xf_max)

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Save in Excel format

wind_solar_proj_2022_no_cpr <- wind_solar_proj_2022_no_cpr |> 
  mutate(theta = theta_no_cpr) 

# Define the path
excel_path_s_w_proj <- file.path(no_cpr_tab_dir, "wind_solar_proj_2022", "01_wind_solar_proj2022_nocpr.xlsx")

# Create workbook
wb_nocpr <- createWorkbook()

# Add first sheet: wind_solar_proj_2022_no_cpr
addWorksheet(wb_nocpr, "Wind_Solar_Projects")
freezePane(wb_nocpr, "Wind_Solar_Projects", firstActiveRow = 2, firstActiveCol = 2)
writeData(wb_nocpr, sheet = "Wind_Solar_Projects", x = wind_solar_proj_2022_no_cpr)

# Add second sheet: equilibrium_prices_no_ratio
addWorksheet(wb_nocpr, "Equilibrium_P_Q")
freezePane(wb_nocpr, "Equilibrium_P_Q", firstActiveRow = 2, firstActiveCol = 2)
writeData(wb_nocpr, sheet = "Equilibrium_P_Q", x = equilibrium_nocpr)

# Save workbook
saveWorkbook(wb_nocpr, file = excel_path_s_w_proj, overwrite = TRUE)

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #


# Section 4: Baseline with CPR --------------------------------------------

# INSTRUCTIONS! --------------------------------------------------------- #
# We will now modeling markets with Countertyparty risk (CPR)
# Baseline results

# We will use the profits function Pi_S(f, gamma, c) in the paper

# Sellers get the following utilities:

#  i.  \Pi_S(c; f; z) =z \int^f_0 p \phi(p) dp + f[1-\Phi(f)z] - R(f,z) - c,
#  ii. r_i = r_0 Var(pxq_i) = r_0(xq_i)^2 Var(p)


#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

#  ---------------------------------------------------------------------- #

# Several functions

# Define several functions that are useful

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# 1. Function to compute the first and second moments of tilde_p

compute_tilde_p_stats <- function(f, alpha, beta) {
  # --- Purpose ---
  # Compute the first moment (mean) and variance of a Beta-distributed variable,
  # where the variable p ~ Beta(alpha, beta) is truncated at value f.
  # That is, tilde_p = min(p, f). The moments of tilde_p are:
    
    #   E[tilde_p] = ∫₀ᶠ p * Beta(p) dp + f * P(p > f)
    #   E[tilde_p^2] = ∫₀ᶠ p² * Beta(p) dp + f² * P(p > f)
  
  # Compute the cumulative probability up to the truncation threshold f ---
  # This gives the total probability mass from 0 to f
  cdf_f <- pbeta(f, alpha, beta)
  
  # Compute the tail probability beyond the truncation threshold f ---
  # This is the probability that p > f
  one_minus_cdf <- 1 - cdf_f
  
  # Compute incomplete Beta function values for moment calculations ---
  # These represent the cumulative probabilities under adjusted Beta distributions,
  # which arise from integrating p * Beta(p) and p^2 * Beta(p)
  
  # I1 corresponds to the integral of p * Beta(p), which behaves like Beta(alpha + 1, beta)
  I1 <- pbeta(f, alpha + 1, beta)
  
  # I2 corresponds to the integral of p^2 * Beta(p), which behaves like Beta(alpha + 2, beta)
  I2 <- pbeta(f, alpha + 2, beta)
  
  # Compute the expected value (first moment) of tilde_p ---
  # Formula:
  #   E[tilde_p] = (E[p] for p in [0, f]) + f * P(p > f)
  #              = (α / (α + β)) * I1 + f * (1 - CDF(f))
  

  first_moment <- (alpha / (alpha + beta)) * I1 + f * one_minus_cdf
  # Same as:
  # first_moment <- integrate(function(p) p * beta_pdf(p),
  #                           lower = 0, upper = f)$value +
  #   f * (1 - beta_cdf(f))
  
  # Compute the second moment of tilde_p ---
  # Formula:
  #   E[tilde_p^2] = (E[p^2] for p in [0, f]) + f² * P(p > f)
  #                = (α(α+1) / ((α+β)(α+β+1))) * I2 + f² * (1 - CDF(f))
  second_moment <- (alpha * (alpha + 1)) / ((alpha + beta) * (alpha + beta + 1)) * I2 +
    f^2 * one_minus_cdf
  
  # Compute the variance of tilde_p ---
  # Var[tilde_p] = E[tilde_p^2] - (E[tilde_p])^2
  var_tilde_p <- second_moment - first_moment^2
  
  # --- Return a named list containing the results ---
  return(list(E_tilde_p = first_moment, Var_tilde_p = var_tilde_p))
}



# 2. Compute R(f, \gamma) with compute_tilde_p_stats

# Function to compute R_i(f, gamma)
compute_R_value_gamma <- function(f, gamma, q_i, x, r_0, alpha, beta) {
  # Compute the tilde_p statistics
  stats <- compute_tilde_p_stats(f, alpha, beta)
  E_tilde_p <- stats$E_tilde_p
  Var_tilde_p <- stats$Var_tilde_p
  
  # Compute R_i(f, gamma) using the given formula
  R_value_gamma <- r_0 * gamma * (q_i^2) * (x^2) *
    (Var_tilde_p + (1 - gamma) * (f - E_tilde_p)^2)
  
  return(R_value_gamma)
}

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Define a function for profits Pi_S, without T (or with T = 0)
# We put T = 0, as we are in our baseline results. We also do this to avoid 
# repetitions of similar functions. We have one that is manageable.

Pi_S_general <- function(f, q_i, x, gamma, r_0, alpha, beta, total_costs, T_values) {
  # Compute ∫₀ᶠ p φ(p) dp numerically
  integral_result <- integrate(function(p) p * dbeta(p, alpha, beta),
                               lower = 0, upper = f)$value
  
  # Compute Φ(f)
  Phi_f <- pbeta(f, alpha, beta)
  
  # Compute R_i(f, gamma)
  R_value_gamma <- compute_R_value_gamma(f, gamma, q_i, x, r_0, alpha, beta)
  
  # Final profit
  profit <- q_i * x * (gamma * integral_result + f * (1 - gamma * Phi_f)) -
    R_value_gamma - total_costs + (T_values * q_i)
  
  return(profit)
}


#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Define the spot market profit function

Pi_S0 <- function(q_i, x, expected_p, total_cost, r) {
  # Compute revenue from spot market sales
  revenue <- q_i * x * expected_p
  
  # Compute total profit by subtracting costs
  profit <- revenue - total_cost - r
  
  return(profit)
}


#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Vectorized version
vectorized_pi_s <- Vectorize(Pi_S_general)


#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# --- Why we use a grid search before uniroot() ---
  # The profit function Pi_S(f) is not guaranteed to be monotonic.
  # It may have multiple roots or no root at all.
  # uniroot() only works if we supply an interval [a, b] where
  # the function changes sign — i.e., f(a) * f(b) < 0.
  # To ensure this, we evaluate the function on a grid and
  # look for intervals with a sign change before calling uniroot().

find_sign_change_index <- function(values, nth = 1) {
  signs <- sign(values)
  change_locs <- which(diff(signs) != 0)
  if (length(change_locs) < nth) return(NA_integer_)
  return(change_locs[nth])
}

# Root where Pi_S_general crosses zero
find_f_root <- function(q_i, x, gamma, r_0, alpha, beta, total_costs, T_values,
                        f_min = 0, f_max = 1, n = 100, tol = 1e-20) {
  f_grid <- seq(f_min, f_max, length.out = n)
  
  profits <- vapply(
    f_grid,
    function(f) Pi_S_general(f, q_i, x, gamma, r_0, alpha, beta, total_costs, T_values),
    numeric(1)
  )
  
  if (max(profits, na.rm = TRUE) < 0) return(NA_real_)
  
  i <- find_sign_change_index(profits, nth = 1)
  if (is.na(i)) return(NA_real_)
  
  root <- uniroot(
    function(f) Pi_S_general(f, q_i, x, gamma, r_0, alpha, beta, total_costs, T_values),
    lower = f_grid[i], upper = f_grid[i + 1], tol = tol
  )$root
  
  return(root)
}



#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

find_f_spot_root <- function(q_i, x, gamma, r_0, alpha, beta, total_costs,
                             expected_p, r, T_values,
                             f_min = 0, f_max = 1, n = 100, tol = 1e-20) {
  f_grid <- seq(f_min, f_max, length.out = n)
  
  pi_s0_val <- Pi_S0(q_i, x, expected_p, total_costs, r)
  
  g_vals <- vapply(
    f_grid,
    function(f) Pi_S_general(f, q_i, x, gamma, r_0, alpha, beta, total_costs, T_values) - pi_s0_val,
    numeric(1)
  )
  
  i <- find_sign_change_index(g_vals, nth = 1)
  if (is.na(i)) return(NA_real_)
  
  root <- uniroot(
    function(f) Pi_S_general(f, q_i, x, gamma, r_0, alpha, beta, total_costs, T_values) - pi_s0_val,
    lower = f_grid[i], upper = f_grid[i + 1], tol = tol
  )$root
  
  return(root)
}

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

find_upper_root <- function(q_i, x, gamma, r_0, alpha, beta, total_costs, T_values,
                            f_min = 0, f_max = 1, n = 100, tol = 1e-20) {
  f_grid <- seq(f_min, f_max, length.out = n)
  
  profits <- vapply(
    f_grid,
    function(f) Pi_S_general(f, q_i, x, gamma, r_0, alpha, beta, total_costs, T_values),
    numeric(1)
  )
  
  i <- find_sign_change_index(profits, nth = 2)
  if (is.na(i)) return(NA_real_)
  
  root <- uniroot(
    function(f) Pi_S_general(f, q_i, x, gamma, r_0, alpha, beta, total_costs, T_values),
    lower = f_grid[i], upper = f_grid[i + 1], tol = tol
  )$root
  
  return(root)
}

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Find the f_c_cpr and f_spot_cpr for each plant
# We then choose the maximum value for each plant between those two values

wind_solar_proj_2022_long <- wind_solar_proj_2022 |> 
  crossing(gamma = gamma_values) |>   # create a line for which gamma defined above
  rowwise() |> 
  mutate(
    # f_c_cpr computation
    f_c_cpr = coalesce(
      find_f_root(q_i = q_i_mwh,
                          x = x,
                          gamma = gamma,
                          r_0 = r_0,
                          alpha = alpha,
                          beta = beta,
                          total_costs = total_cost,
                          T_values = 0),
      1),

    # f_spot_cpr: value for f for which Pi_S_general - Pi_S0 = 0
    f_spot_cpr = coalesce(
      find_f_spot_root(q_i = q_i_mwh,
                       x = x,
                       gamma = gamma,
                       r_0 = r_0,
                       alpha = alpha,
                       beta = beta,
                       total_costs = total_cost,
                       expected_p = expected_p,
                       r = r,
                       T_values = 0),
      0),
  # f_upper_cpr : for \Pi_S_general that cross \Pi_S = 0 two times, find the second root
  f_upper = 
    find_upper_root(q_i = q_i_mwh,
                             x = x,
                             gamma = gamma,
                             r_0 = r_0,
                             alpha = alpha,
                             beta = beta,
                             total_costs = total_cost,
                             T_values = 0)
  ) |> 
  ungroup() |> 
      mutate(f_upper_message = if_else(
        !is.na(f_upper),
        if_else(f_upper > expected_p, "f_upper > E(p)", "f_upper <= E(p)"),
        NA_character_
      ),
      f_max_cpr = pmax(f_c_cpr, f_spot_cpr, na.rm = TRUE),
             xf_c_cpr = x * f_c_cpr,
             xf_spot_cpr = x * f_spot_cpr,
             xf_upper_cpr = x * f_upper,
             xf_max_cpr = x * f_max_cpr
             ) 

# Create of cumulative capacity and production

wind_solar_proj_2022_long <- wind_solar_proj_2022_long |> 
  arrange(gamma, xf_max_cpr) |> 
  group_by(gamma) |> 
  mutate(cumulative_production = cumsum(q_i_mwh),
         cumulative_capacity = cumsum(capacity),
         x_q_exp_p_total_costs = x * expected_p * q_i_mwh - total_cost
  ) |>
  ungroup()


# We compute R_f_max_cpr_gamma. If we replace in the profits function Pi_S_general with
# f_max_cpr_gamma and R_f_max_cpr_gamma, the function Pi_S_general should be 0 for each line

# wind_solar_proj_2022_long_baseline <- wind_solar_proj_2022_long |> 
#   rowwise() |> 
#   mutate(R_f_max_cpr_gamma = compute_R_value_gamma(f_max_cpr, gamma, q_i_mwh, x, r_0, alpha, beta)) |> 
#   relocate(R_f_max_cpr_gamma, .after = gamma)


wind_solar_proj_2022_long_baseline <- crossing(
  wind_solar_proj_2022_long,
  theta = theta_values
) |> 
  arrange(theta, xf_max_cpr, gamma)




# We use a local to not overwriting other possibly existing values
# This is just a manuel check to see if roots are well-computed

# Define parameters for the Francisco Pizarro Solar Farm 12
local({
  # Parameters for the example
  q_i <- 1911908
  x <- 60
  gamma <- 0.5
  r_0 <- get("r_0", envir = .GlobalEnv)  # retrieve from global env
  alpha <- get("alpha", envir = .GlobalEnv)
  beta <- get("beta", envir = .GlobalEnv)
  total_costs <- 46263720
  T_values <- 0
  
  # Generate profits
  f_grid <- seq(0, 1, length.out = 200)
  profits <- sapply(f_grid, function(f) Pi_S_general(f, q_i, x, gamma, r_0, alpha, beta, total_costs, T_values))
  df_plot <- data.frame(f = f_grid, profit = profits)
  
  # Create plot
  p <- ggplot(df_plot, aes(x = f, y = profit)) +
    geom_line(color = "steelblue") +
    geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
    labs(
      x = "f",
      y = expression("Profit (" * Pi[S]^Baseline * ")")
    ) +
    theme_minimal(base_size = 28)
  
  print(p)

})


#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Compute contract supply using xf_max_cpr

# Compute contract supply using xf_max_cpr
contract_supply_cpr <- wind_solar_proj_2022_long_baseline |> 
  mutate(gamma = as.factor(gamma)) |>  # Ensure gamma is a factor for grouping
  group_by(gamma, xf_max_cpr, theta) |>  
  summarise(
    total_capacity = sum(capacity, na.rm = TRUE),
    total_production = sum(q_i_mwh, na.rm = TRUE)
  ) |> 
  ungroup() |> 
  arrange(theta, gamma, xf_max_cpr) |>  
  group_by(gamma, theta) |>  
  mutate(
    cumulative_capacity = cumsum(total_capacity),
    cumulative_production = cumsum(total_production),
    q_0 = q_0,
    G_expected_p_x = cumsum(if_else(xf_max_cpr <= expected_p * x, total_capacity, 0)),
    gamma_num = as.numeric(as.character(gamma))  # Numeric for gradient coloring
  ) |> 
  mutate(
    G_expected_p_x = last(G_expected_p_x, order_by = xf_max_cpr)
  ) |>   
  ungroup()

# Plot loop by theta
for (i in seq_along(theta_values)) {
  theta_val <- theta_values[i]
  index_label <- sprintf("%02d", i)
  
  # Demand curve (vertical line)
  demand_segments <- tibble(
    x_start = theta_val,
    x_end   = theta_val,
    y_start = threshold_price,
    y_end   = min(contract_supply_cpr$xf_max_cpr)
  )
  
  # Filter for supply curve and create plot
  supply_cpr <- ggplot(
    contract_supply_cpr |> 
      filter(xf_max_cpr <= expected_p * x),
    aes(
      x = cumulative_capacity,
      y = xf_max_cpr,
      group = gamma,         # Keeps step-line structure per gamma
      color = gamma_num      # Uses gradient color
    )
  ) +
    geom_step() +
    geom_point(size = 0.5) +
    geom_segment(
      data = demand_segments,
      aes(x = x_start, xend = x_end, y = y_start, yend = y_end),
      color = "black", linetype = "solid", size = 1,
      inherit.aes = FALSE
    ) +
    scale_color_gradientn(
      colours = theme_palette_gamma,
      name = expression(gamma)
    ) +
    labs(
      x = "Cumulative Capacity (MW)",
      y = "Contract Price (€/MWh)"
    ) +
    theme_minimal(base_size = base_s) +
    theme(
      legend.position = c(0.05, 0.95),  # Top-left inside plot (x, y from 0 to 1)
      legend.justification = c(0, 1),  # Anchor top-left corner of the legend box
      legend.background = element_rect(
        fill = alpha("white", 0.2),  # Semi-transparent white background
        color = NA                           # No border
      ),
      legend.title = element_text(face = "bold"),
      panel.grid.major = element_line(color = "grey90", size = 0.2),
      panel.grid.minor = element_line(color = "grey95", size = 0.1)
    )
  
  print(supply_cpr)
  
  plot_filename <- paste0(index_label, "_supply_function_cpr_theta_", theta_val, ".pdf")
  plot_path_cpr <- file.path(baseline_fig_path, plot_filename)
  
  ggsave(plot_path_cpr, plot = supply_cpr, width = 16, height = 9, dpi = 300)
  
  message("✅ Saved plot for theta = ", theta_val, " at: ", plot_path_cpr)
}


#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Equilibrium Prices Dataset Creation

# Pre-process the data once
wind_solar_proj_2022_long_baseline <- wind_solar_proj_2022_long_baseline %>%
  arrange(gamma, cumulative_capacity)

# Compute equilibrium prices across theta values
equilibrium_prices <- purrr::map_dfr(theta_values, function(theta_val) {
  
  # Get the last row *before* theta (for each gamma), with price < threshold
  before_theta <- wind_solar_proj_2022_long_baseline %>%
    group_by(gamma) %>%
    filter(cumulative_capacity < theta_val, xf_max_cpr < threshold_price) %>%
    slice_tail(n = 1) %>%
    ungroup()
  
  # Get the first row *at or after* theta (for each gamma)
  after_theta <- wind_solar_proj_2022_long_baseline %>%
    filter(cumulative_capacity >= theta_val) %>%
    group_by(gamma) %>%
    slice_head(n = 1) %>%
    select(gamma, next_price = xf_max_cpr) %>%
    ungroup()
  
  # Join and determine equilibrium price logic
  before_theta %>%
    left_join(after_theta, by = "gamma") %>%
    mutate(
      equilibrium_price = if_else(
        !is.na(next_price) & next_price >= threshold_price,
        threshold_price,
        next_price
      ),
      equilibrium_quantity = cumulative_capacity,
      theta = theta_val
    ) %>%
    select(gamma, equilibrium_quantity, equilibrium_price, theta)
})

# Sort for plotting
equilibrium_prices <- equilibrium_prices %>%
  arrange(theta, gamma)

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Welfare Dataset Creation

wind_solar_proj_2022_long_baseline <- wind_solar_proj_2022_long_baseline |> 
  left_join(
    equilibrium_prices |> 
      select(gamma, theta, equilibrium_price, equilibrium_quantity),
    by = c("gamma", "theta")
  ) 

wind_solar_proj_2022_long_baseline <- wind_solar_proj_2022_long_baseline |> 
  mutate(f_equilibrium = equilibrium_price / x,
         xf_equilibrium = equilibrium_price) |> 
  select(-equilibrium_price)


wind_solar_proj_2022_long_baseline <- wind_solar_proj_2022_long_baseline |>
  arrange(theta, gamma, xf_max_cpr) |>
  group_by(gamma) |> 
  mutate(
    R_f_equilibrium_cpr = compute_R_value_gamma(
      f_equilibrium, gamma, q_i_mwh, x, r_0, alpha, beta
    )
  ) |> 
  ungroup() |>
  mutate(
    x_q_exp_p_total_costs_R = x * expected_p * q_i_mwh - total_cost - R_f_equilibrium_cpr
  )


wind_solar_proj_2022_long_baseline <- wind_solar_proj_2022_long_baseline |>
  mutate(
    profit_cpr_contracts = vectorized_pi_s(
      f = f_equilibrium,
      q_i = q_i_mwh,
      x = x,
      gamma = gamma,
      r_0 = r_0,
      alpha = alpha,
      beta = beta,
      total_costs = total_cost,
      T_values = 0
    )
  )


ordered_vars <- c(
  "theta", "gamma", "f_c_cpr", "f_spot_cpr", "f_upper", "f_upper_message",
  "f_max_cpr", "f_equilibrium", "R_f_equilibrium_cpr",
  "xf_c_cpr", "xf_spot_cpr", "xf_upper_cpr", "xf_max_cpr", "xf_equilibrium",
  "equilibrium_quantity", "x_q_exp_p_total_costs", "x_q_exp_p_total_costs_R",
  "cumulative_production", "cumulative_capacity", "profit_cpr_contracts"
)

# Reorder dataframe
wind_solar_proj_2022_long_baseline <- wind_solar_proj_2022_long_baseline |>
  relocate(
    profits_sp_no_cpr, .after = last_col()
  ) |>
  relocate(all_of(ordered_vars), .after = profits_sp_no_cpr)

# STEP 1: Prepare list of unique gamma values
gammas <- wind_solar_proj_2022_long_baseline |>
  distinct(gamma) |>
  arrange(gamma)

# STEP 2: Compute W^0 (baseline welfare) for gamma = 0
W_0 <- wind_solar_proj_2022_long_baseline |>
  filter(gamma == 0, profits_sp_no_cpr >= 0) |>
  group_by(theta) |>
  summarise(
    W_0 = sum(x_q_exp_p_total_costs - r, na.rm = TRUE),
    .groups = "drop"
  ) |> 
  ungroup()

# STEP 3: Compute W(gamma): welfare under contracts for each gamma
W_gamma <- wind_solar_proj_2022_long_baseline |>
  group_by(gamma, theta) |>
  summarise(
    welfare_gamma_eur = sum(
      if_else(xf_max_cpr <= xf_equilibrium & cumulative_capacity < theta,
              x_q_exp_p_total_costs_R, 0),
      na.rm = TRUE
    ),
    .groups = "drop"
  ) |> 
  ungroup()

# STEP 4: Extract W(gamma) at gamma = 0 to use as reference in ratios
W_gamma_0 <- W_gamma |>
  filter(gamma == 0) |>
  select(theta, welfare_gamma_ref_eur = welfare_gamma_eur)

# STEP 5: Extract equilibrium quantity and price for gamma = 0
equilibrium_gamma_0 <- equilibrium_prices |>
  filter(gamma == 0) |>
  select(
    theta,
    eq_quantity_gamma_0 = equilibrium_quantity,
    eq_price_gamma_0 = equilibrium_price
  )

# STEP 6: Assemble final dataset
welfare_dataset_baseline <- equilibrium_prices |>
  left_join(W_gamma, by = c("gamma", "theta")) |>
  left_join(W_gamma_0, by = "theta") |>
  left_join(W_0, by = "theta") |>
  left_join(equilibrium_gamma_0, by = "theta") |>
  mutate(
    eq_price = equilibrium_price,
    eq_quantity = equilibrium_quantity,
    
    welfare_ratio_percent = (welfare_gamma_eur / welfare_gamma_ref_eur) * 100,
    welfare_gap_million_eur = (welfare_gamma_ref_eur - welfare_gamma_eur) / 1e6,
    welfare_gain_million_eur = (welfare_gamma_eur - W_0) / 1e6,
    
    eq_quantity_ratio_percent = (eq_quantity / eq_quantity_gamma_0) * 100,
    eq_price_ratio_percent = (eq_price / eq_price_gamma_0) * 100
  ) |>
  select(
    gamma, theta,
    eq_price, eq_price_ratio_percent,
    eq_quantity, eq_quantity_ratio_percent,
    W_0, welfare_gamma_eur, welfare_ratio_percent,
    welfare_gap_million_eur, welfare_gain_million_eur
  ) |>
  arrange(theta, gamma)

welfare_dataset_baseline


#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Profits by gamma and theta

# Step 1: Compute seller profits under contracts
profits_by_gamma_theta <- wind_solar_proj_2022_long_baseline |>
  filter(xf_max_cpr <= xf_equilibrium & cumulative_capacity < theta) |>
  group_by(gamma, theta) |>
  summarise(
    seller_profits_eur = sum(profit_cpr_contracts, na.rm = TRUE),
    .groups = "drop"
  ) |>
  arrange(theta, gamma)

# Step 2: Join with existing welfare dataset
welfare_with_profits_baseline <- welfare_dataset_baseline |>
  left_join(profits_by_gamma_theta, by = c("gamma", "theta"))

# Step 3: Compute profit shares for sellers and buyers
welfare_with_profits_baseline <- welfare_with_profits_baseline |>
  mutate(
    buyer_profits_eur = welfare_gamma_eur - seller_profits_eur,
    seller_profit_share_percent = round((seller_profits_eur / welfare_gamma_eur) * 100, 2),
    buyer_profit_share_percent  = round((buyer_profits_eur / welfare_gamma_eur) * 100, 2)
  ) |>
  select(
    gamma, theta,
    welfare_gamma_eur,
    seller_profits_eur,
    buyer_profits_eur,
    seller_profit_share_percent,
    buyer_profit_share_percent
  ) |>
  arrange(theta, gamma)


#  ---------------------------------------------------------------------- #
# Plot functions

# Define a reusable plot function
plot_profit_metric <- function(data,
                               x_var,
                               y_var,
                               group_var,
                               color_var = group_var,
                               colors = theme_palette_theta,
                               x_label = NULL,
                               y_label = NULL,
                               color_label = NULL,
                               y_scale_million = FALSE,
                               y_scale_percent = FALSE,
                               save_path,
                               file_name,
                               legend_position = "bottom") {  
  if (y_scale_million && y_scale_percent) stop("Cannot use both million and percent scaling")
  
  if (is.null(x_label)) x_label <- x_var
  if (is.null(y_label)) y_label <- y_var
  if (is.null(color_label)) color_label <- color_var
  
  # Dynamically determine shapes
  n_groups <- length(unique(data[[color_var]]))
  available_shapes <- 0:25
  if (n_groups > length(available_shapes)) {
    stop("Too many groups for shapes (max is 26). Reduce groups or use faceting.")
  }
  shape_vals <- available_shapes[1:n_groups]
  
  plot <- ggplot(data, aes(
    x = .data[[x_var]],
    y = .data[[y_var]],
    group = .data[[group_var]],
    color = as.factor(.data[[color_var]]),
    shape = as.factor(.data[[color_var]])
  )) +
    geom_line(linewidth = 1) +
    geom_point(size = 3) +
    scale_color_manual(values = colors, name = color_label) +
    scale_shape_manual(values = shape_vals, name = color_label) +
    labs(x = x_label, y = y_label) +
    theme_minimal(base_size = base_s) +
    theme(
      legend.position = legend_position,
      legend.direction = "vertical",
      legend.box.background = element_blank(),  # No border box
      legend.background = element_blank(),      # No fill
      legend.title = element_text(face = "bold"),
      panel.grid.major = element_line(color = "grey90", size = 0.2),
      panel.grid.minor = element_line(color = "grey95", size = 0.1)
    ) +
    guides(color = guide_legend(ncol = 1), shape = guide_legend(ncol = 1))
  
  if (y_scale_million) {
    plot <- plot + scale_y_continuous(labels = function(x) paste0(scales::comma(x / 1e6, accuracy = 1), "M"))
  }
  if (y_scale_percent) {
    plot <- plot + scale_y_continuous(labels = function(x) paste0(x))
  }
  
  if (!dir.exists(save_path)) {
    dir.create(save_path, recursive = TRUE, showWarnings = FALSE)
  }
  
  ggsave(file.path(save_path, file_name),
         plot = plot,
         width = 16,
         height = 9,
         dpi = 300)
  
  return(plot)
}


plot_line_by_gamma <- function(data,
                               x,
                               y,
                               group_var = "gamma",
                               color_var = group_var,
                               color_label = color_var,
                               y_lab = NULL,
                               x_lab = "T",
                               title = NULL,
                               y_scale_million = FALSE,
                               y_scale_comma = FALSE,
                               filename = NULL,
                               folder = NULL,
                               base_size = base_s,
                               colors = theme_palette_theta,
                               fill_points = FALSE,
                               legend_position = "bottom") { 
  
  n_groups <- length(unique(data[[color_var]]))
  available_shapes <- 0:25
  if (n_groups > length(available_shapes)) {
    stop("Too many groups for shapes (max is 26). Reduce groups or use faceting.")
  }
  shape_vals <- available_shapes[1:n_groups]
  
  p <- ggplot(data, aes(
    x = .data[[x]],
    y = .data[[y]],
    group = .data[[group_var]],
    color = as.factor(.data[[color_var]]),
    shape = as.factor(.data[[color_var]])
  )) +
    geom_line(linewidth = 1) +
    geom_point(size = 3) +
    scale_color_manual(values = colors, name = color_label) +
    scale_shape_manual(values = shape_vals, name = color_label) +
    labs(
      x = x_lab,
      y = y_lab %||% y,
      title = title
    ) +
    theme_minimal(base_size = base_size) +
    theme(
      legend.position = legend_position,
      legend.direction = "vertical",
      legend.box.background = element_blank(),  # No border box
      legend.background = element_blank(),      # No fill
      legend.title = element_text(face = "bold"),
      panel.grid.major = element_line(color = "grey90", size = 0.2),
      panel.grid.minor = element_line(color = "grey95", size = 0.1)
    ) +
    guides(color = guide_legend(ncol = 1), shape = guide_legend(ncol = 1))
  
  
  if (y_scale_million) {
    p <- p + scale_y_continuous(labels = function(x) paste0(scales::comma(x / 1e6, accuracy = 1), "M"))
  } else if (y_scale_comma) {
    p <- p + scale_y_continuous(labels = scales::comma)
  }
  
  if (!is.null(filename) && !is.null(folder)) {
    path <- file.path(folder, filename)
    ggsave(path, plot = p, width = 16, height = 9, dpi = 300)
    message("✅ Saved plot to: ", path)
  }
  
  print(p)
  invisible(p)
}




# Plots ----------------------------------------------------------------- #
# Call the function for both seller and buyer profits

# Equilibrium Prices
eq_prices <- plot_line_by_gamma(
  data            = equilibrium_prices,
  x               = "gamma",
  y               = "equilibrium_price",
  group_var       = "theta",
  color_var       = "theta",
  color_label     = expression(theta),
  y_lab           = "Contract Price (€/MW)",
  x_lab           = expression(gamma),
  fill_points     = TRUE,
  filename        = "04_equilibrium_price_vs_gamma_theta.pdf",
  folder          = baseline_fig_path,
  legend_position = c(0.08, 0.90) 
)

# Equilibrium Quantities
eq_quantities <- plot_line_by_gamma(
  data = welfare_dataset_baseline,
  x = "gamma",
  y = "eq_quantity",
  group_var   = "theta",
  color_var   = "theta",
  color_label = expression(theta),
  y_lab = "Investment (MW)",
  x_lab = expression(gamma),
  fill_points = TRUE,  # Added to enable shape+fill
  filename = "05_equilibrium_quantity_vs_gamma_theta.pdf",
  folder = baseline_fig_path,
  legend_position = c(0.90, 0.90) 
)

# Welfare
welfare_plot <- plot_line_by_gamma(
  data = welfare_dataset_baseline,
  x = "gamma",
  y = "welfare_gamma_eur",
  group_var   = "theta",
  color_var   = "theta",
  color_label = expression(theta),
  y_lab = "Welfare (M€)",
  x_lab = expression(gamma),
  y_scale_million = TRUE,
  fill_points = TRUE,  # Added for consistency
  filename = "06_welfare_vs_gamma_theta.pdf",
  folder = baseline_fig_path,
  legend_position = c(0.90, 0.90) 
)

# Seller Profit (in Millions)
plot_profit_metric(
  data             = welfare_with_profits_baseline,
  x_var            = "gamma",
  x_label          = expression(gamma),
  y_var            = "seller_profits_eur",
  group_var        = "theta",
  color_var        = "theta",
  color_label      = expression(theta),
  y_label          = "Seller Profit (M€)",
  y_scale_million  = TRUE,
  save_path        = baseline_fig_path,
  file_name        = "07_seller_profit_vs_gamma_theta.pdf",
  legend_position  = c(0.08, 0.90)  # ⬅️ Legend top-left inside plot
)


# Buyer Profit (in Millions)
plot_profit_metric(
  data             = welfare_with_profits_baseline,
  x_var            = "gamma",
  x_label          = expression(gamma),
  y_var            = "buyer_profits_eur",
  group_var        = "theta",
  color_var        = "theta",
  color_label      = expression(theta),
  y_label          = "Buyer Profit (M€)",
  y_scale_million  = TRUE,
  save_path        = baseline_fig_path,
  file_name        = "08_buyer_profit_vs_gamma_theta.pdf",
  legend_position  = c(0.08, 0.1)  
)

# Seller Profit Share (%)
plot_profit_metric(
  data        = welfare_with_profits_baseline,
  x_var       = "gamma",
  x_label     = expression(gamma),
  y_var       = "seller_profit_share_percent",
  y_label     = "Seller Profit Share (%)",
  group_var   = "theta",
  color_var   = "theta",
  color_label = expression(theta),  
  y_scale_percent = TRUE,
  save_path   = baseline_fig_path,
  file_name   = "09_seller_profit_share_vs_gamma_theta.pdf",
  legend_position  = c(0.08, 0.90)  
)

# Buyer Profit Share (%)
plot_profit_metric(
  data        = welfare_with_profits_baseline,
  x_var       = "gamma",
  x_label     = expression(gamma),
  y_var       = "buyer_profit_share_percent",
  group_var   = "theta",
  color_var   = "theta",
  color_label = expression(theta),  
  y_label     = "Buyer Profit Share (%)",
  y_scale_percent = TRUE,
  save_path   = baseline_fig_path,
  file_name   = "10_buyer_profit_share_vs_gamma_theta.pdf",
  legend_position  = c(0.08, 0.1)  
)


#  ---------------------------------------------------------------------- #

# Save Baseline file
# Baseline file

filter_by_gamma <- function(data) {
  data %>% filter(gamma %in% selected_gammas)
}

datasets <- list(df1 = wind_solar_proj_2022_long_baseline, 
                 df2 = equilibrium_prices, 
                 df3 = welfare_dataset_baseline,
                 df4 = welfare_with_profits_baseline)

filtered_datasets <- lapply(datasets, filter_by_gamma)


# Construct dynamic filename
excel_filename_base <- paste0("01_wind_solar_projects_cpr_theta", ".xlsx")
output_path_base <- file.path(baseline_path, excel_filename_base)

# Create workbook
wb_base <- createWorkbook()

# Define sheet names in same order as filtered_datasets
sheet_names <- c("Wind Solar Projects", 
                 "Equilibrium P. & Q.", 
                 "Welfare", 
                 "Profits")

# Loop through datasets and sheet names
for (i in seq_along(filtered_datasets)) {
  addWorksheet(wb_base, sheet_names[i])
  freezePane(wb_base, sheet_names[i], firstActiveRow = 2, firstActiveCol = 2)
  writeData(wb_base, sheet = sheet_names[i], x = filtered_datasets[[i]])
}

# Save workbook
saveWorkbook(wb_base, file = output_path_base, overwrite = TRUE)


#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Section 5: Public Guarantees --------------------------------------------

# Welfare with Public Guarantees

f_equilibrium_gamma_0_df <- wind_solar_proj_2022_long_baseline |> 
  filter(gamma == 0) |> 
  distinct(theta, f_equilibrium) |> 
  mutate(xf_equilibrium = x * f_equilibrium)

 # Results and data for gamma = 0 
gamma_0_data <- wind_solar_proj_2022_long_baseline |> 
  filter(gamma == 0) |> 
  select(-gamma)

# Repeat gamma_0_data for each gamma
public_guarantees <- crossing(
  gamma = wind_solar_proj_2022_long_baseline$gamma,
  gamma_0_data
) |> 
  relocate(gamma, .after = theta)

equilibrium_public_g <- public_guarantees |>
distinct(gamma, theta, xf_equilibrium, equilibrium_quantity) |>
arrange(gamma)
# 
# #  ---------------------------------------------------------------------- #
# # Function to obtain the integral
# 
# Define integrand: (f* - p) x beta_pdf(p)

compute_integral_vec <- function(f_vec, alpha, beta) {
  vapply(f_vec, function(f) {
    integrate(function(p) (f - p) * dbeta(p, alpha, beta), lower = 0, upper = f)$value
  }, numeric(1))
}

# #  ---------------------------------------------------------------------- #

public_guarantees <- public_guarantees |>
  mutate(
    x_integral = x * compute_integral_vec(f_equilibrium, alpha, beta),
    q_i_mwh_lambda_gamma_x_integral = q_i_mwh * lambda * gamma * x_integral
  ) |> 
  arrange(theta, gamma, xf_max_cpr) 



# STEP 1: W_0 (P.G. welfare under gamma = 0)
W_0_pg <- public_guarantees |>
  filter(gamma == 0, profits_sp_no_cpr >= 0) |>
  group_by(theta) |>
  summarise(
    welfare_0_eur = sum(x_q_exp_p_total_costs - r, na.rm = TRUE),
    .groups = "drop"
  )

# STEP 2: W_gamma for all gamma-theta combinations (adjusted for guarantees)
W_gamma_pg <- public_guarantees |>
  filter(xf_max_cpr <= xf_equilibrium & cumulative_capacity < theta) |>
  group_by(gamma, theta) |>
  summarise(
    x_q_exp_p_total_costs_R         = sum(x_q_exp_p_total_costs_R, na.rm = TRUE),
    q_i_mwh_lambda_gamma_x_integral = sum(q_i_mwh_lambda_gamma_x_integral, na.rm = TRUE),
    welfare_gamma_eur = x_q_exp_p_total_costs_R - q_i_mwh_lambda_gamma_x_integral,
    .groups = "drop"
  ) |> 
  ungroup()

# STEP 3: Baseline gamma = 0 (P.G.) values for comparison
W_gamma_0_pg <- W_gamma_pg |>
  filter(gamma == 0) |>
  select(theta, welfare_gamma_0_eur = welfare_gamma_eur)


# STEP 4: Equilibrium quantities/prices at gamma = 0 (for ratio comparisons)
eq_gamma_0_pg <- equilibrium_public_g |>
  filter(gamma == 0) |>
  select(
    theta,
    equilibrium_quantity_gamma_0 = equilibrium_quantity,
    equilibrium_price_gamma_0 = xf_equilibrium
  )

# Combine all metrics into one final table

welfare_dataset_pg <- equilibrium_public_g |>
  left_join(W_gamma_pg,    by = c("gamma", "theta")) |>
  left_join(W_gamma_0_pg,  by = "theta") |>
  left_join(W_0_pg,        by = "theta") |>
  left_join(eq_gamma_0_pg, by = "theta") |>
  mutate(
    # Recompute key equilibrium metrics
    equilibrium_price_eur     = xf_equilibrium,
    equilibrium_quantity_mw   = equilibrium_quantity,
    
    # Show W_0 only for gamma == 0, retain full for calc
    W_0 = if_else(gamma == 0, welfare_0_eur, NA_real_),
    
    # Welfare analysis
    welfare_gain_vs_baseline_meur = (welfare_gamma_eur - welfare_0_eur) / 1e6,
    welfare_ratio_percent         = (welfare_gamma_eur / welfare_gamma_0_eur) * 100,
    welfare_gap_million_eur       = (welfare_gamma_0_eur - welfare_gamma_eur) / 1e6,
    
    # Quantity & price ratios vs gamma = 0
    eq_quantity_ratio_percent = (equilibrium_quantity / equilibrium_quantity_gamma_0) * 100,
    eq_price_ratio_percent    = (xf_equilibrium / equilibrium_price_gamma_0) * 100
  ) |>
  select(
    gamma, theta,
    equilibrium_price_eur,
    eq_price_ratio_percent,
    equilibrium_quantity_mw,
    eq_quantity_ratio_percent,
    W_0,
    welfare_gamma_eur,
    welfare_ratio_percent,
    welfare_gap_million_eur,
    welfare_gain_vs_baseline_meur
  ) |>
  arrange(theta, gamma)

welfare_dataset_pg


welfare_dataset_baseline <- welfare_dataset_baseline %>%
  select(gamma, theta, welfare_gamma_eur) |> 
  rename(welfare_gamma_eur_baseline = welfare_gamma_eur)

welfare_ratios_pg <- W_gamma_pg |> 
  left_join(welfare_dataset_baseline, by = c("gamma", "theta")) |> 
  mutate(
    welfare_ratio = (welfare_gamma_eur / welfare_gamma_eur_baseline) * 100
  ) |> 
  select(theta, gamma, everything()) |> 
  arrange(gamma, theta)

welfare_ratios_wide_pg <- welfare_ratios_pg |> 
  select(gamma, theta, welfare_ratio) |> 
  pivot_wider(
    names_from = theta,
    values_from = welfare_ratio,
    names_prefix = "theta_"
  )

#  ---------------------------------------------------------------------- #
# Profits under Public Guarantees

profits_by_gamma_theta_pg <- public_guarantees |> 
  filter(xf_max_cpr <= xf_equilibrium & cumulative_capacity < theta) |> 
  group_by(gamma, theta) |> 
  summarise(
    seller_profits_eur_pg = sum(profit_cpr_contracts, na.rm = TRUE),
    .groups = "drop"
  ) |> 
  ungroup() |> 
  arrange(theta, gamma, seller_profits_eur_pg)

welfare_dataset_pg <- welfare_dataset_pg |> 
  left_join(profits_by_gamma_theta_pg, by = c("gamma", "theta")) |> 
  mutate(buyer_profits_eur_pg = welfare_gamma_eur - seller_profits_eur_pg,
         buyer_profit_share_percent = round((buyer_profits_eur_pg / welfare_gamma_eur) * 100, 2), 
         seller_profit_share_percent = round((seller_profits_eur_pg / welfare_gamma_eur) * 100, 2),
         ) |> 
  arrange(theta, gamma)


profits_by_gamma_theta_pg <- welfare_dataset_pg |> 
  select(theta, gamma, starts_with("buyer"), starts_with("seller"))

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #
# Plots for Public Guarantees


# Contract Prices

# Equilibrium Price Plot
plot_line_by_gamma(
  data        = welfare_dataset_pg,
  x           = "gamma",
  y           = "equilibrium_price_eur",
  group_var   = "theta",
  color_var   = "theta",
  color_label = expression(theta),
  y_lab       = "Contract Price (€/MW)",
  x_lab       = expression(gamma),
  colors      = theme_palette_theta,
  filename    = "01_equilibrium_price_vs_gamma_theta_publicg.pdf",
  folder      = with_pg_fig_path,
  legend_position = c(0.08, 0.85) 
)

# Equilibrium Quantity Plot
plot_line_by_gamma(
  data        = welfare_dataset_pg,
  x           = "gamma",
  y           = "equilibrium_quantity_mw",
  group_var   = "theta",
  color_var   = "theta",
  color_label = expression(theta),
  y_lab       = "Investment (MW)",
  x_lab       = expression(gamma),
  colors      = theme_palette_theta,
  filename    = "02_equilibrium_quantity_vs_gamma_theta_publicg.pdf",
  folder      = with_pg_fig_path,
  legend_position = c(0.08, 0.85)
)

# Welfare Plot (in million €)
plot_line_by_gamma(
  data            = welfare_dataset_pg,
  x               = "gamma",
  y               = "welfare_gamma_eur",
  group_var       = "theta",
  color_var       = "theta",
  color_label     = expression(theta),
  y_lab           = "Welfare (M€)",
  x_lab           = expression(gamma),
  y_scale_million = TRUE,
  colors          = theme_palette_theta,
  filename        = "03_welfare_vs_gamma_theta_publicg.pdf",
  folder          = with_pg_fig_path,
  legend_position = c(0.08, 0.85)
)

# --------------- Profit Metrics ---------------- #

# Seller Profit (M€)
plot_profit_metric(
  data            = welfare_dataset_pg,
  x_var           = "gamma",
  x_label         = expression(gamma),
  y_var           = "seller_profits_eur_pg",
  group_var       = "theta",
  color_var       = "theta",
  color_label     = expression(theta),    
  y_label         = "Seller Profit (M€)",
  y_scale_million = TRUE,
  save_path       = with_pg_fig_path,
  file_name       = "04_seller_profit_vs_gamma_theta_publicg.pdf",
  legend_position = c(0.08, 0.85)
)

# Buyer Profit ( M€)
plot_profit_metric(
  data            = welfare_dataset_pg,
  x_var           = "gamma",
  y_var           = "buyer_profits_eur_pg",
  group_var       = "theta",
  color_var       = "theta",
  color_label     = expression(theta),    
  y_label         = "Buyer Profit (M€)",
  x_lab           = expression(gamma),
  y_scale_million = TRUE,
  save_path       = with_pg_fig_path,
  file_name       = "05_buyer_profit_vs_gamma_theta_publicg.pdf",
  legend_position = c(0.08, 0.70)
)

# Seller Profit Share (%)
plot_profit_metric(
  data              = welfare_dataset_pg,
  x_var             = "gamma",
  y_var             = "seller_profit_share_percent",
  group_var         = "theta",
  color_var         = "theta",
  color_label       = expression(theta),  
  y_label           = "Seller Profit Share (%)",
  x_lab             = expression(gamma),  
  y_scale_percent   = TRUE,
  save_path         = with_pg_fig_path,
  file_name         = "06_seller_profit_share_vs_gamma_theta_publicg.pdf",
  legend_position = c(0.08, 0.85)
)

# Buyer Profit Share (%)
plot_profit_metric(
  data              = welfare_dataset_pg,
  x_var             = "gamma",
  y_var             = "buyer_profit_share_percent",
  group_var         = "theta",
  color_var         = "theta",
  color_label       = expression(theta),    
  y_label           = "Buyer Profit Share (%)",
  x_lab             = expression(gamma),  
  y_scale_percent   = TRUE,
  save_path         = with_pg_fig_path,
  file_name         = "07_buyer_profit_share_vs_gamma_theta_publicg.pdf",
  legend_position = c(0.08, 0.70)
)

#  ---------------------------------------------------------------------- #

# Save Public Guarantees

# Define dataset list for public guarantees
datasets_guarantees <- list(
  `Wind Solar Projects (Public G.)` = public_guarantees,
  `Equilibrium P. & Q.`             = equilibrium_public_g,
  `Welfare Public G.`               = welfare_dataset_pg,
  `Welfare Baseline`                = welfare_dataset_baseline,
  `Welfare Ratios Pub.G.`           = welfare_ratios_pg,
  `Welfare Ratios Pub.G. (Wide)`    = welfare_ratios_wide_pg,
  `Profits (P.G.)`                  = profits_by_gamma_theta_pg
)

# Filter all datasets by gamma
filtered_guarantees <- lapply(datasets_guarantees, filter_by_gamma)

# Construct filename and output path
excel_filename_guarantees <- "01_wind_solar_projects_cpr_theta_with_public-g.xlsx"
output_path_guarantees <- file.path(with_public_guarantees_path, excel_filename_guarantees)

# Create workbook
wb_guarantees <- createWorkbook()

# Add sheets dynamically
for (sheet in names(filtered_guarantees)) {
  addWorksheet(wb_guarantees, sheet)
  freezePane(wb_guarantees, sheet, firstActiveRow = 2, firstActiveCol = 2)
  writeData(wb_guarantees, sheet = sheet, x = filtered_guarantees[[sheet]])
}

# Save workbook
saveWorkbook(wb_guarantees, file = output_path_guarantees, overwrite = TRUE)

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Section 6: Public Subsidies ---------------------------------------------

# INSTRUCTIONS! --------------------------------------------------------- #
# We will now modeling markets with CPR and add public subsidies T

# We will use the profits function (1) in the paper

# Sellers get the following utilities:

#  i.  \Pi_S_general(c; f; z) =z \int^f_0 p \phi(p) dp + f[1-\Phi(f)z] - R(f,z) - c + T*q_i,
#  ii. r_i = r_0 Var(pxq_i) = r_0(xq_i)^2 Var(p)

#  iii. Be careful! r_0 should be sufficiently low, in order to 
#  make sure that some plants find it optimal to trade in spot m.
#  for at least some plants, we should have \Pi_S^0(c) > 0.

# Function for further plotting

plot_profit_share_vs_T <- function(data, 
                                   y_var, 
                                   y_label, 
                                   file_name, 
                                   save_path, 
                                   y_scale_million = FALSE,
                                   y_scale_percent = FALSE,
                                   base_size = base_s,
                                   base_palette = base_palette,
                                   colors = NULL,
                                   legend_position = "bottom") {  # ⬅️ new argument
  
  # Unique gamma groups
  gamma_levels <- sort(unique(data$gamma))
  n_groups <- length(gamma_levels)
  
  # Dynamic color assignment if not provided
  if (is.null(colors)) {
    colors <- gradient_n_pal(base_palette)(seq(0, 1, length.out = n_groups))
  }
  
  # Determine shapes
  filled_shapes <- 21:25
  available_shapes <- 0:25
  
  if (n_groups <= length(filled_shapes)) {
    shape_vals <- filled_shapes[1:n_groups]
    shape_mode <- "filled"
  } else {
    shape_vals <- available_shapes[1:n_groups]
    shape_mode <- "unfilled"
  }
  
  # Base plot
  plot <- ggplot(data, aes(
    x = T_values,
    y = .data[[y_var]],
    group = gamma,
    color = as.factor(gamma),
    shape = as.factor(gamma)
  )) +
    geom_line(linewidth = 1)
  
  # Add points
  if (shape_mode == "filled") {
    plot <- plot +
      geom_point(aes(fill = as.factor(gamma)), size = 2, stroke = 0.6) +
      scale_fill_manual(values = colors, name = expression(gamma))
  } else {
    plot <- plot + geom_point(size = 4)
  }
  
  # Scales and labels
  plot <- plot +
    scale_color_manual(values = colors, name = expression(gamma)) +
    scale_shape_manual(values = shape_vals, name = expression(gamma)) +
    labs(
      x = "T",
      y = y_label
    ) +
    theme_minimal(base_size = base_size) +
    theme(
      legend.position = legend_position,
      legend.title = element_text(face = "bold"),
      legend.box = if (identical(legend_position, "bottom")) "vertical" else NULL,
      panel.grid.major = element_line(color = "grey90", size = 0.2),
      panel.grid.minor = element_line(color = "grey95", size = 0.1)
    ) +
    guides(
      color = guide_legend(ncol = 1),
      shape = guide_legend(ncol = 1)
    )
  
  # Y-axis formatting
  if (y_scale_million) {
    plot <- plot + scale_y_continuous(
      labels = function(x) paste0(scales::comma(x / 1e6), "M")
    )
  }
  
  if (y_scale_percent) {
    plot <- plot + scale_y_continuous(
      labels = function(x) paste0(format(round(x, 1), nsmall = 0, trim = TRUE))
    )
  }
  
  # Show and save plot
  print(plot)
  
  ggsave(
    filename = file.path(save_path, file_name),
    plot     = plot,
    width    = 16,
    height   = 9,
    dpi      = 300
  )
  
  message("✅ Saved: ", file_name)
}



#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Set your T values

# Set up parallel processing
plan(multisession, workers = availableCores() - 1)


# Split sample case
T_val <- seq(0, 0.05, by = 0.01)  

# Improved version using functional programming principles
wind_solar_proj_2022_T <- wind_solar_proj_2022 %>%
  crossing(gamma = gamma_values, 
           T_values = T_val, 
           theta = theta_values) |> 
  mutate(
    f_c_cpr = future_pmap_dbl(
      list(q_i_mwh, x, gamma, r_0, alpha, beta, total_cost, T_values),
      ~ coalesce(
        find_f_root(
          ..1, ..2, ..3, ..4, ..5, ..6, ..7, ..8
        ),
        1  # Default value
      )
    ),
    
    f_spot_cpr = future_pmap_dbl(
      list(q_i_mwh, x, gamma, r_0, alpha, beta, total_cost, expected_p, r, T_values),
      ~ coalesce(
        find_f_spot_root(
          ..1, ..2, ..3, ..4, ..5, ..6, ..7, ..8, ..9, ..10
        ),
        0  # Default value
      )
    ),
    f_upper = future_pmap_dbl(
      list(q_i_mwh, x, gamma, r_0, alpha, beta, total_cost, T_values),
      ~ find_upper_root(
          ..1, ..2, ..3, ..4, ..5, ..6, ..7, ..8
        )
      )
    ) %>%
  # Post-processing in vectorized operations
  mutate(
    f_max_cpr = pmax(f_c_cpr, f_spot_cpr, na.rm = TRUE),
    across(c(f_c_cpr, f_spot_cpr), ~ coalesce(., NA_real_))
  )

wind_solar_proj_2022_T <- wind_solar_proj_2022_T |> 
  mutate(
    f_upper_message = if_else(
      !is.na(f_upper),
      if_else(f_upper > expected_p, "f_upper > E(p)", "f_upper <= E(p)"),
      NA_character_),
    xf_c_cpr = x * f_c_cpr,
    xf_spot_cpr = x * f_spot_cpr,
    xf_upper_cpr = x * f_upper,
    xf_max_cpr = x * f_max_cpr
  )


wind_solar_proj_2022_T <- wind_solar_proj_2022_T |>
  arrange(theta, T_values, gamma, xf_max_cpr) |>
  group_by(theta, gamma, T_values) |>
  mutate(cumulative_production = cumsum(q_i_mwh),
         cumulative_capacity = cumsum(capacity)
         ) |> 
  ungroup() |> 
  mutate(x_q_exp_p_total_costs = x * expected_p * q_i_mwh - total_cost,
         lambda = lambda,
         lambda_q_i_mwh_T = lambda * T_values * q_i_mwh
  )
#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #
# Contract Supply


for (i in seq_along(T_val)) {
  T_value <- T_val[i]
  
  # Filter for current T slice
  wind_T_data <- wind_solar_proj_2022_T |> 
    filter(T_values == T_value)
  
  # Get unique theta values
  theta_values <- sort(unique(wind_T_data$theta))
  
  for (j in seq_along(theta_values)) {
    theta_val <- theta_values[j]
    
    # Filter for this (T, theta)
    slice_data <- wind_T_data |> 
      filter(theta == theta_val) |> 
      mutate(
        gamma = as.factor(gamma),
        gamma_num = as.numeric(as.character(gamma))
      )
    
    # Demand line
    demand_segments <- tibble(
      x_start = theta_val,
      x_end   = theta_val,
      y_start = threshold_price,
      y_end   = min(slice_data$xf_max_cpr, na.rm = TRUE)
    )
    
    # Plot
    supply_plot <- ggplot(
      slice_data |> filter(xf_max_cpr <= expected_p * x),
      aes(
        x = cumulative_capacity,
        y = xf_max_cpr,
        group = gamma,
        color = gamma_num
      )
    ) +
      geom_step() +
      geom_point(size = 0.5) +
      geom_segment(
        data = demand_segments,
        aes(x = x_start, xend = x_end, y = y_start, yend = y_end),
        color = "black", linetype = "solid", size = 1,
        inherit.aes = FALSE
      ) +
      scale_color_gradientn(
        colours = theme_palette_gamma,
        name = expression(gamma)
      ) +
      labs(
        x = "Cumulative Capacity (MW)",
        y = "Contract Price (€/MWh)"
      ) +
      theme_minimal(base_size = base_s) +
      theme(
        legend.position = c(0.05, 0.95),
        legend.justification = c(0, 1),
        legend.background = element_rect(fill = alpha("white", 0.2), color = NA),
        legend.title = element_text(face = "bold"),
        panel.grid.major = element_line(color = "grey90", size = 0.2),
        panel.grid.minor = element_line(color = "grey95", size = 0.1)
      )
    
    # Labels for filename
    theta_label <- format(round(theta_val), big.mark = "")
    T_label <- ifelse(
      T_value < 1,
      formatC(T_value, format = "f", digits = 3),
      as.character(T_value)
    )
    
    filename <- paste0(
      sprintf("%02d", i),
      "_supply_function_cpr_T_", T_label,
      "_theta_", theta_label, ".pdf"
    )
    
    plot_path <- file.path(with_T_fig_path, filename)
    
    ggsave(plot_path, plot = supply_plot, width = 16, height = 9, dpi = 300)
    
    message("✅ Saved plot for T = ", T_label, ", theta = ", theta_label, " at: ", plot_path)
  }
}

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Compute Welfare

# Equilibrium Prices Dataset Creation

  # 1. For each gamma, get the last row before theta_val (preserving cumulative capacity)
  before_theta_T <- wind_solar_proj_2022_T |> 
    group_by(gamma, T_values, theta) |> 
    arrange(cumulative_capacity) |> 
    filter(cumulative_capacity < theta, xf_max_cpr < threshold_price) |> 
    slice_tail(n = 1) |> 
    ungroup()
  
  # 2. For each gamma, get the first row at or after theta_val (to peek at the next price)
  after_theta_T <- wind_solar_proj_2022_T |> 
    group_by(gamma, T_values, theta) |> 
    arrange(cumulative_capacity) |> 
    filter(cumulative_capacity >= theta) |> 
    slice_head(n = 1) |> 
    ungroup() |> 
    select(gamma, next_price = xf_max_cpr, T_values, theta)
  
  # 3. Join the two and update the price:
  #    - Keep the cumulative capacity from before_theta
  #    - Update the price to threshold_price only if the next row's price is >= threshold_price.
  equilibrium_prices_T <- before_theta_T |> 
    left_join(after_theta_T, by = c("gamma", "T_values", "theta")) |> 
    mutate(
      equilibrium_price = if_else(
        !is.na(next_price) & next_price >= threshold_price,
        threshold_price,
        next_price
      ),
      equilibrium_quantity = cumulative_capacity
    ) %>%
    select(gamma, theta, T_values, equilibrium_quantity, equilibrium_price) |> 
    arrange(theta, T_values, gamma)
  
# View the combined results
print(equilibrium_prices_T)

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Equilibrium Prices Dataset + Some reordering

wind_solar_proj_2022_T <- wind_solar_proj_2022_T |> 
  left_join(
    equilibrium_prices_T |> 
      select(gamma, theta, T_values, equilibrium_price, equilibrium_quantity),
    by = c("gamma", "T_values", "theta")
  ) 

wind_solar_proj_2022_T <- wind_solar_proj_2022_T |>
  mutate(f_equilibrium = equilibrium_price / x,
         xf_equilibrium = equilibrium_price) |> 
  select(-equilibrium_price)


wind_solar_proj_2022_T <- wind_solar_proj_2022_T |>
  arrange(theta, T_values, gamma, xf_max_cpr) |>
  rowwise() |> 
  mutate(R_f_equilibrium_cpr = 
           compute_R_value_gamma(f_equilibrium, gamma, q_i_mwh, x, r_0, alpha, beta)) |> 
  mutate(
    x_q_exp_p_total_costs_R = x * expected_p * q_i_mwh - total_cost
    - R_f_equilibrium_cpr
  )

wind_solar_proj_2022_T <- wind_solar_proj_2022_T |> 
  mutate(
    profit_cpr_contracts = 
      vectorized_pi_s(
        f = f_equilibrium,
        q_i = q_i_mwh,
        x = x,
        gamma = gamma,
        r_0 = r_0,
        alpha = alpha,
        beta = beta,
        total_costs = total_cost,
        T_values = T_values
      )
  )


ordered_vars <- c(
  "theta", "gamma", "lambda", "lambda_q_i_mwh_T", "T_values", "f_c_cpr", "f_spot_cpr",
  "f_upper", "f_upper_message", "f_max_cpr", "f_equilibrium", 
  "R_f_equilibrium_cpr", "xf_c_cpr", "xf_spot_cpr", "xf_upper_cpr", "xf_max_cpr", 
  "xf_equilibrium", "equilibrium_quantity", "x_q_exp_p_total_costs", "x_q_exp_p_total_costs_R",
  "cumulative_production", "cumulative_capacity", "profit_cpr_contracts"
)

# Reorder dataframe
wind_solar_proj_2022_T <- wind_solar_proj_2022_T |> 
  select(
    everything(),              # Keeps all variables in current order...
    -all_of(ordered_vars),     # ...but temporarily removes the ones you want to reorder
    -profits_sp_no_cpr,                # temporarily remove profits_sp_no_cpr to place it explicitly
    profits_sp_no_cpr,                 # put profits_sp_no_cpr in place
    all_of(ordered_vars)       # then add ordered vars in the desired order
  )


#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Welfare Dataset Creation

# STEP 1: W_0 (welfare under gamma = 0, profits spot > 0)
W_0_T <- wind_solar_proj_2022_T |>
  filter(gamma == 0, profits_sp_no_cpr >= 0) |>
  group_by(T_values, theta) |>
  summarise(
    welfare_0_eur = sum(x_q_exp_p_total_costs - r, na.rm = TRUE),
    .groups = "drop"
  ) |> 
  ungroup()

# STEP 2: W_gamma for all gamma-theta combinations (adjusted for guarantees)
W_gamma_T <- wind_solar_proj_2022_T |>
  group_by(gamma, T_values, theta) |> 
  filter(xf_max_cpr <= xf_equilibrium & cumulative_capacity < theta) |> 
  summarise(
    x_q_exp_p_total_costs_R = sum(
      x_q_exp_p_total_costs_R
    ),
    lambda_T_q = sum(lambda * T_values * q_i_mwh),
    welfare_gamma_eur = x_q_exp_p_total_costs_R - lambda_T_q,
    .groups = "drop",
    theta = first(theta)
  ) |> 
  ungroup()


# STEP 3: Baseline gamma = 0 values for comparison
W_gamma_0_T <- W_gamma_T |>
  group_by(theta, T_values) |> 
  filter(gamma == 0) |>
  select(theta, gamma, T_values, welfare_gamma_0_eur = welfare_gamma_eur) |> 
  ungroup()


# STEP 4: Equilibrium quantities/prices at gamma = 0 (for ratio comparisons)
eq_gamma_0_T <- equilibrium_prices_T |>
  group_by(theta, T_values) |> 
  filter(gamma == 0) |>
  select(
    theta,
    gamma,
    T_values,
    equilibrium_quantity_gamma_0 = equilibrium_quantity,
    equilibrium_price_gamma_0 = equilibrium_price
  ) |> 
    ungroup()



# Combine all metrics into one final table

welfare_dataset_T_full <- equilibrium_prices_T |>
  left_join(W_gamma_T, by = c("theta", "gamma", "T_values")) |>
  
  left_join(
    W_gamma_0_T |> select(theta, T_values, welfare_gamma_0_eur),
    by = c("theta", "T_values")
  ) |>
  
  left_join(W_0_T, by = c("theta", "T_values")) |>
  
  left_join(eq_gamma_0_T |> select(-gamma), by = c("theta", "T_values")) |>
  
  mutate(
    equilibrium_price_eur     = equilibrium_price,
    equilibrium_quantity_mw   = equilibrium_quantity,
    
    W_0 = if_else(gamma == 0, welfare_0_eur, NA_real_),
    
    welfare_gain_vs_baseline_meur = (welfare_gamma_eur - welfare_0_eur) / 1e6,
    welfare_ratio_percent         = (welfare_gamma_eur / welfare_gamma_0_eur) * 100,
    welfare_gap_million_eur       = (welfare_gamma_0_eur - welfare_gamma_eur) / 1e6,
    
    eq_quantity_ratio_percent = (equilibrium_quantity / equilibrium_quantity_gamma_0) * 100,
    eq_price_ratio_percent    = (equilibrium_price / equilibrium_price_gamma_0) * 100
  ) |>
  
  select(
    theta, gamma, T_values,
    equilibrium_price_eur,
    eq_price_ratio_percent,
    equilibrium_quantity_mw,
    eq_quantity_ratio_percent,
    W_0,
    welfare_gamma_eur,
    welfare_ratio_percent,
    welfare_gap_million_eur,
    welfare_gain_vs_baseline_meur
  ) |>
  
  arrange(T_values, gamma)



#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #
# for subsequent graphs/tables, we only loot at theta = 3,500

# Compute W(gamma) for each gamma

welfare_ratios_filtered <- W_gamma_T |> 
  filter(theta == selected_theta) |>
  left_join(welfare_dataset_baseline, by = c("gamma", "theta")) |> 
  mutate(
    welfare_ratio = (welfare_gamma_eur / welfare_gamma_eur_baseline) * 100
    ) |> 
  select(theta, gamma, T_values, everything()) |> 
  arrange(gamma, T_values)

welfare_ratios_wide_T <- welfare_ratios_filtered |> 
  select(T_values, gamma, theta, welfare_ratio) |> 
  pivot_wider(
    names_from = T_values,
    values_from = welfare_ratio,
    names_prefix = "T_val_"
  )


#  ---------------------------------------------------------------------- #
# Profits under T

profits_by_gamma_theta_T <- wind_solar_proj_2022_T |>
  group_by(theta, gamma, T_values) |> 
  filter(xf_max_cpr <= xf_equilibrium & cumulative_capacity < theta) |> 
  summarise(
    seller_profits_eur_T = sum(profit_cpr_contracts, na.rm = TRUE),
    .groups = "drop",
    theta = first(theta)
  ) |> 
  ungroup() |> 
  arrange(theta, gamma, T_values, seller_profits_eur_T)


welfare_T_with_profits <- welfare_dataset_T_full |>
  left_join(profits_by_gamma_theta_T, by = c("gamma", "theta", "T_values"))

# Step 3: Compute profit shares for sellers and buyers
welfare_T_with_profits <- welfare_T_with_profits |>
  mutate(
    buyer_profits_eur_T = welfare_gamma_eur - seller_profits_eur_T,
    seller_profit_T_share_percent = round((seller_profits_eur_T / welfare_gamma_eur) * 100, 2),
    buyer_profit_T_share_percent  = round((buyer_profits_eur_T / welfare_gamma_eur) * 100, 2)
  ) |>
  select(
    gamma, theta, T_values,
    welfare_gamma_eur,
    seller_profits_eur_T,
    buyer_profits_eur_T,
    seller_profit_T_share_percent,
    buyer_profit_T_share_percent
  ) |>
  arrange(theta, gamma)


#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Plots

# We select a fewer gammas and only for selected_theta
# to avoid multiple lines in plots

# Filter datasets
equilibrium_prices_T_filtered <- equilibrium_prices_T |>
  filter(gamma %in% selected_gammas,
         theta %in% selected_theta)

welfare_dataset_T_filtered <- welfare_dataset_T_full |>
  filter(gamma %in% selected_gammas,
         theta %in% selected_theta)

welfare_T_with_profits_filtered <- welfare_T_with_profits |>
  filter(gamma %in% selected_gammas,
         theta %in% selected_theta)


wind_solar_proj_2022_T_filtered <- wind_solar_proj_2022_T |> 
  filter(gamma %in% selected_gammas,
         theta %in% selected_theta)


theme_palette_gamma_selected <- gradient_n_pal(base_palette)(
  seq(0, 1, length.out = length(selected_gammas))
)

# Prices
plot_eq_price <- plot_line_by_gamma(
  data        = equilibrium_prices_T_filtered,
  x           = "T_values",
  y           = "equilibrium_price",
  group_var   = "gamma",
  color_var   = "gamma",
  color_label = expression(gamma),
  x_lab       = expression(T),
  y_lab       = "Contract Price (€/MW)",
  colors      = theme_palette_gamma_selected,
  filename    = "06_equilibrium_price_vs_gamma_T.pdf",
  folder      = with_T_fig_path,
  legend_position = c(0.05, 0.70)
)

# Quantities
plot_eq_quantities <- plot_line_by_gamma(
  data          = equilibrium_prices_T_filtered,
  x             = "T_values",
  y             = "equilibrium_quantity",
  group_var     = "gamma",
  color_var     = "gamma",
  color_label   = expression(gamma),
  x_lab         = expression(T),
  y_lab         = "Investment (MW)",
  y_scale_comma = TRUE,
  colors        = theme_palette_gamma_selected,
  filename      = "07_equilibrium_quantity_vs_gamma_T.pdf",
  folder        = with_T_fig_path,
  legend_position = c(0.05, 0.70)
)

# Welfare
plot_welfare <- plot_line_by_gamma(
  data            = welfare_dataset_T_filtered,
  x               = "T_values",
  y               = "welfare_gamma_eur",
  group_var       = "gamma",
  color_var       = "gamma",
  color_label     = expression(gamma),
  x_lab           = expression(T),
  y_lab           = "Welfare (M€)",
  y_scale_million = TRUE,
  colors          = theme_palette_gamma_selected,
  filename        = "08_equilibrium_welfare_vs_gamma_T.pdf",
  folder          = with_T_fig_path,
  legend_position = c(0.05, 0.30)
)

#  ---------------------------------------------------------------------- #

# Profits Plots

plot_profit_share_vs_T(
  data             = welfare_T_with_profits_filtered,
  y_var            = "seller_profits_eur_T",
  y_label          = "Seller Profit (M€)",
  file_name        = "09_seller_profit_vs_gamma_T.pdf",
  save_path        = with_T_fig_path,
  y_scale_million  = TRUE,
  base_palette     = base_palette,
  legend_position = c(0.05, 0.85)
)


plot_profit_share_vs_T(
  data             = welfare_T_with_profits_filtered,
  y_var            = "seller_profit_T_share_percent",
  y_label          = "Seller Profit (%)",
  file_name        = "10_seller_profit_share_vs_gamma_T.pdf",
  save_path        = with_T_fig_path,
  y_scale_percent  = TRUE,
  base_palette     = base_palette,
  legend_position = c(0.05, 0.85)
)


#  ---------------------------------------------------------------------- #

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Save everything: for now, we save only results for theta = 3500

# Define dataset list for T-based results
datasets_T <- list(
  `Wind Solar Projects (T)` = wind_solar_proj_2022_T_filtered,
  `Equilibrium P. & Q.`     = equilibrium_prices_T_filtered,
  `Welfare with T`          = welfare_dataset_T_filtered,
  `Welfare Ratios`          = welfare_ratios_filtered,
  `Profits (with T)`        = welfare_T_with_profits_filtered
)

# Filter all datasets by gamma
filtered_T <- lapply(datasets_T, filter_by_gamma)

# Label theta for filename
theta_label <- format(round(unique(wind_solar_proj_2022_T_filtered$theta)), big.mark = "")

# Construct output path
excel_filename_T <- paste0("01_wind_solar_projects_cpr_T_theta_", theta_label, ".xlsx")
output_path_T <- file.path(with_T_path, excel_filename_T)

# Create workbook
wb_T <- createWorkbook()

# Add sheets dynamically
for (sheet in names(filtered_T)) {
  addWorksheet(wb_T, sheet)
  freezePane(wb_T, sheet, firstActiveRow = 2, firstActiveCol = 2)
  writeData(wb_T, sheet = sheet, x = filtered_T[[sheet]])
}

# Save workbook
saveWorkbook(wb_T, file = output_path_T, overwrite = TRUE)


#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Section 7: Public G. and Public Subsidies Comparison --------------------

# find the T* that makes public expenditure under G same as public expenditure under T,
# for every plant.

#  ---------------------------------------------------------------------- #
# Function
find_T_star_for_gamma_theta <- function(gamma_val, theta_val, df, lambda) {
  # Filter for the given gamma and theta
  df_sub <- df |> 
    filter(gamma == gamma_val, theta == theta_val)
  
  # Return NA if no data
  if (nrow(df_sub) == 0) {
    return(tibble(gamma = gamma_val, theta = theta_val, T_star = NA_real_))
  }
  
  # Compute the root
  root <- uniroot(
    f = function(T_val) {
      sum_q <- sum(df_sub$q_i_mwh, na.rm = TRUE)
      sum_q_lambda <- sum(df_sub$q_i_mwh_lambda_gamma_x_integral)
      (T_val * sum_q * lambda) - sum_q_lambda
    },
    interval = c(0, 100), # a big interval just in case
    tol = 1e-6
  )
  
  return(tibble(gamma = gamma_val, 
                theta = theta_val, 
                T_star = root$root))
}

#  ---------------------------------------------------------------------- #

# Extract the T_star_values and some checks

# Get all combinations from your dataset
gamma_theta_grid <- public_guarantees |> 
  distinct(gamma, theta)

# Map the function across all rows
T_star_results <- gamma_theta_grid |>
  mutate(
    T_star = purrr::map2_dbl(gamma, theta, 
                             ~ find_T_star_for_gamma_theta(.x, .y, public_guarantees, lambda)$T_star)
  )

# This is just a dataset to check if lambda * T_star * q_i_mwh = q_i_mwh_lambda_gamma_x_integral
# They should be equal

check_public_expenditures <- public_guarantees |> 
  left_join(T_star_results, by = c("gamma", "theta")) |> 
  mutate(
    lhs = lambda * T_star * q_i_mwh,
    rhs = q_i_mwh_lambda_gamma_x_integral,
    diff = lhs - rhs
  ) |> 
  arrange(theta, gamma, xf_c_cpr)



# Contains exactly the same values. We are good!
check_public_expend_summary <- check_public_expenditures |> 
  filter(xf_max_cpr <= xf_equilibrium & cumulative_capacity < theta) |> 
  group_by(gamma, theta) |> 
  reframe(
    lhs = lambda * first(T_star) * sum(q_i_mwh, na.rm = TRUE),
    rhs = sum(q_i_mwh_lambda_gamma_x_integral, na.rm = TRUE),
    # show that lhs (i.e.) lambda * T_star * q_i_mwh = q_i_mwh_lambda_gamma_x_integral
    # so the equal_within_tolerance should be TRUE for all (1e-7 = 0.0000001)
    equal_within_tolerance = abs(lhs - rhs) < 1e-7
  )

#  ---------------------------------------------------------------------- #


results_with_T_G_comparison <- wind_solar_proj_2022_long |> 
  crossing(theta = theta_values) |> 
  left_join(T_star_results, by = c("gamma", "theta")) |> 
  mutate(
    f_c_cpr = future_pmap_dbl(
      list(q_i_mwh, x, gamma, r_0, alpha, beta, total_cost, T_star),
      ~ coalesce(
        find_f_root(
          ..1, ..2, ..3, ..4, ..5, ..6, ..7, ..8
        ),
        1  # Default value
      )
    ),
    
    f_spot_cpr = future_pmap_dbl(
      list(q_i_mwh, x, gamma, r_0, alpha, beta, total_cost, expected_p, r, T_star),
      ~ coalesce(
        find_f_spot_root(
          ..1, ..2, ..3, ..4, ..5, ..6, ..7, ..8, ..9, ..10
        ),
        0  # Default value
      )
    ),
    f_upper = future_pmap_dbl(
      list(q_i_mwh, x, gamma, r_0, alpha, beta, total_cost, T_star),
      ~ find_upper_root(
          ..1, ..2, ..3, ..4, ..5, ..6, ..7, ..8
        )
      )
    ) |> 
  mutate(
    f_max_cpr = pmax(f_c_cpr, f_spot_cpr, na.rm = TRUE),
    xf_c_cpr = x * f_c_cpr,
    xf_spot_cpr = x * f_spot_cpr,
    xf_max_cpr = x * f_max_cpr
  )

results_with_T_G_comparison <- results_with_T_G_comparison |> 
  arrange(theta, gamma, xf_max_cpr) |>
  group_by(gamma, theta, T_star) |>
  mutate(cumulative_production = cumsum(q_i_mwh),
         cumulative_capacity = cumsum(capacity),
         f_upper_message = if_else(!is.na(f_upper),
           if_else(f_upper > expected_p, "f_upper > E(p)", "f_upper <= E(p)"),
           NA_character_
         )
  ) |> 
  ungroup() |> 
  mutate(lambda = lambda,
         lambda_q_i_mwh_T = lambda * T_star * q_i_mwh
         ) 


# We will merge the results of the x_integral_by_theta with our main dataset

x_integral_by_theta <- public_guarantees |>
  distinct(theta, f_equilibrium) |>
  mutate(
    x_integral = x * compute_integral_vec(f_equilibrium, alpha = alpha, beta = beta)
  ) |>
  select(theta, f_equilibrium, x_integral)


results_with_T_G_comparison <- results_with_T_G_comparison |> 
  left_join(x_integral_by_theta, by = "theta") |> 
  mutate(q_i_mwh_lambda_gamma_x_integral = q_i_mwh * lambda * gamma * x_integral)


#  ---------------------------------------------------------------------- #

# Compute Welfare

# Equilibrium Prices Dataset Creation

# 1. For each gamma, get the last row before theta_val (preserving cumulative capacity)
before_theta_T <- results_with_T_G_comparison |> 
  group_by(gamma, theta, T_star) |> 
  arrange(cumulative_capacity) |> 
  filter(cumulative_capacity < theta, xf_max_cpr < threshold_price) |> 
  slice_tail(n = 1) |> 
  ungroup()

# 2. For each gamma, get the first row at or after theta_val (to peek at the next price)
after_theta_T <- results_with_T_G_comparison |> 
  group_by(gamma, theta) |> 
  arrange(cumulative_capacity) |> 
  filter(cumulative_capacity >= theta) |> 
  slice_head(n = 1) |> 
  ungroup() |> 
  select(gamma, T_star, theta, next_price = xf_max_cpr, T_star, theta)

# 3. Join the two and update the price:
#    - Keep the cumulative capacity from before_theta
#    - Update the price to threshold_price only if the next row's price is >= threshold_price.
equilibrium_prices_T <- before_theta_T |> 
  left_join(after_theta_T, by = c("gamma", "T_star", "theta")) |> 
  mutate(
    equilibrium_price = if_else(
      !is.na(next_price) & next_price >= threshold_price,
      threshold_price,
      next_price
    ),
    equilibrium_quantity = cumulative_capacity
  )  |> 
  select(gamma, T_star, theta, equilibrium_quantity, equilibrium_price) 

# View the combined results
print(equilibrium_prices_T)

#  ---------------------------------------------------------------------- #

# Equilibrium Prices Dataset + Some reordering

results_with_T_G_comparison <- results_with_T_G_comparison |> 
  left_join(
    equilibrium_prices_T |> 
      select(gamma, theta, T_star, equilibrium_price, equilibrium_quantity),
    by = c("gamma", "T_star", "theta")
  ) 

results_with_T_G_comparison <- results_with_T_G_comparison |>
  mutate(f_equilibrium = equilibrium_price / x,
         xf_equilibrium = equilibrium_price) |> 
  select(-equilibrium_price)


results_with_T_G_comparison <- results_with_T_G_comparison |>
  arrange(theta, gamma, xf_max_cpr) |>
  rowwise() |> 
  mutate(R_f_equilibrium_cpr = 
           compute_R_value_gamma(f_equilibrium, gamma, q_i_mwh, x, r_0, alpha, beta)) |> 
  mutate(
    x_q_exp_p_total_costs_R = x * expected_p * q_i_mwh - total_cost
    - R_f_equilibrium_cpr
  )

results_with_T_G_comparison <- results_with_T_G_comparison |> 
  mutate(
    profit_cpr_contracts = 
      vectorized_pi_s(
        f = f_equilibrium,
        q_i = q_i_mwh,
        x = x,
        gamma = gamma,
        r_0 = r_0,
        alpha = alpha,
        beta = beta,
        total_costs = total_cost,
        T_values = T_star
      )
    )


ordered_vars <- c(
  "theta", "gamma", "lambda", "T_star", "f_c_cpr", "f_spot_cpr",
  "f_max_cpr", "f_equilibrium", "R_f_equilibrium_cpr",
  "xf_c_cpr", "xf_spot_cpr", "xf_max_cpr", "xf_equilibrium",
  "equilibrium_quantity", "x_q_exp_p_total_costs", "x_q_exp_p_total_costs_R",
  "cumulative_production", "cumulative_capacity", "profit_cpr_contracts",
  "x_integral", "lambda_q_i_mwh_T", "q_i_mwh_lambda_gamma_x_integral"
)

# Reorder dataframe
results_with_T_G_comparison <- results_with_T_G_comparison |> 
  select(
    everything(),              # Keeps all variables in current order...
    -all_of(ordered_vars),     # ...but temporarily removes the ones you want to reorder
    -profits_sp_no_cpr,                # temporarily remove profits_sp_no_cpr to place it explicitly
    profits_sp_no_cpr,                 # put profits_sp_no_cpr in place
    all_of(ordered_vars)       # then add ordered vars in the desired order
  )

results_with_T_G_comparison <- results_with_T_G_comparison |> 
  mutate(q_i_mwh_T = q_i_mwh * T_star,
         q_i_mwh_gamma_x_integral = q_i_mwh * gamma * x_integral) |> 
  arrange(theta, gamma, T_star, xf_max_cpr)

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Welfare Dataset Creation

# STEP 1: W_0 (welfare under gamma = 0, profits spot > 0)
W_0_T <- results_with_T_G_comparison |>
  filter(gamma == 0, profits_sp_no_cpr >= 0) |> 
  group_by(theta) |>
  summarise(
    welfare_0_eur = sum(x_q_exp_p_total_costs - r, na.rm = TRUE),
    .groups = "drop"
  )

# STEP 2: W_gamma for all gamma-theta combinations (adjusted for guarantees)
W_gamma_T <- results_with_T_G_comparison |>
  group_by(gamma, theta) |> 
  filter(xf_max_cpr <= xf_equilibrium & cumulative_capacity < theta) |> 
  summarise(
    x_q_exp_p_total_costs_R = sum(
      x_q_exp_p_total_costs_R
    ),
    lambda_T_q = sum(lambda_q_i_mwh_T),
    welfare_gamma_eur = x_q_exp_p_total_costs_R - lambda_T_q,
    .groups = "drop"
  ) |> 
  ungroup() 

# STEP 3: Baseline gamma = 0 values for comparison
W_gamma_0_T <- W_gamma_T |>
  group_by(theta) |> 
  filter(gamma == 0) |>
  select(theta, welfare_gamma_eur_0 = welfare_gamma_eur) |> 
  ungroup()


# STEP 4: Equilibrium quantities/prices at gamma = 0 (for ratio comparisons)
eq_gamma_0_vals <- equilibrium_prices_T |> 
  group_by(theta, T_star) |> 
  filter(gamma == 0) |> 
  ungroup() |> 
  select(theta, 
         equilibrium_quantity_gamma_0 = equilibrium_quantity, 
         equilibrium_price_gamma_0 = equilibrium_price) 



welfare_dataset_G_T <- equilibrium_prices_T |>
  # Step 1: Add welfare for each gamma
  left_join(W_gamma_T, by = c("gamma", "theta")) |> 
  
  # Step 2: Add welfare at gamma = 0 (same theta only!)
  left_join(W_gamma_0_T, by = "theta") |>   
  
  # Step 3: Add baseline W_0 (spot profits > 0 at gamma = 0)
  left_join(W_0_T, by = "theta") |> 
  
  # Step 4: Add equilibrium values at gamma = 0
  left_join(eq_gamma_0_vals, by = "theta") |> 

  
  # Step 5: Compute comparative metrics
  mutate(
    equilibrium_price_eur     = equilibrium_price,
    equilibrium_quantity_mw   = equilibrium_quantity,
    
    W_0 = if_else(gamma == 0, welfare_0_eur, NA_real_),
    
    welfare_gain_vs_baseline_meur = (welfare_gamma_eur - welfare_0_eur) / 1e6,
    welfare_ratio_percent         = (welfare_gamma_eur / welfare_gamma_eur_0) * 100,
    welfare_gap_million_eur       = (welfare_gamma_eur_0 - welfare_gamma_eur) / 1e6,
    
    eq_quantity_ratio_percent     = (equilibrium_quantity / equilibrium_quantity_gamma_0) * 100,
    eq_price_ratio_percent        = (equilibrium_price / equilibrium_price_gamma_0) * 100
  ) |>
  select(
    gamma, theta, T_star, W_0,
    equilibrium_price_eur,
    eq_price_ratio_percent,
    equilibrium_quantity_mw,
    eq_quantity_ratio_percent,
    welfare_gamma_eur,
    welfare_ratio_percent,
    welfare_gap_million_eur,
    welfare_gain_vs_baseline_meur
  ) |>
  arrange(gamma, theta)


welfare_dataset_G_T

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

W_gamma_T <- results_with_T_G_comparison |> 
  group_by(gamma, theta, T_star)  |> 
  filter(xf_max_cpr <= xf_equilibrium & cumulative_capacity < theta) |> 
  summarise(
    x_q_exp_p_total_costs_R = sum(
      x_q_exp_p_total_costs_R
    ),
    lambda_T_q = sum(lambda_q_i_mwh_T),
    welfare_gamma_eur = x_q_exp_p_total_costs_R - lambda_T_q,
    .groups = "drop"
  ) 


pg_welfare <- welfare_dataset_pg |> 
  select(welfare_gamma_eur_pg = welfare_gamma_eur, 
         buyer_profits_eur_pg, theta, gamma)


welfare_ratios <- W_gamma_T |> 
  left_join(pg_welfare, by = c("gamma", "theta")) |> 
  mutate(
    welfare_ratio = (welfare_gamma_eur / welfare_gamma_eur_pg) * 100
  ) |> 
  select(theta, gamma, T_star, everything()) |> 
  arrange(theta, gamma)

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Profits under public expenditure under T = public expenditure under G
profits_by_gamma_theta_T <- results_with_T_G_comparison |> 
  group_by(gamma, theta, T_star) |> 
  filter(xf_max_cpr <= xf_equilibrium & cumulative_capacity < theta) |> 
  summarise(
    seller_profits_eur = sum(profit_cpr_contracts, na.rm = TRUE),
    .groups = "drop"
  ) |> 
  ungroup() |> 
  arrange(gamma, theta, T_star, seller_profits_eur)


welfare_with_profits_G_T <- welfare_dataset_G_T |> 
  left_join(profits_by_gamma_theta_T, by = c("gamma", "theta", "T_star")) 

# Step 3: Compute profit shares for sellers and buyers
welfare_with_profits_G_T <- welfare_with_profits_G_T |>
  mutate(
    buyer_profits_eur = welfare_gamma_eur - seller_profits_eur,
    seller_profit_share_percent = round((seller_profits_eur / welfare_gamma_eur) * 100, 2),
    buyer_profit_share_percent  = round((buyer_profits_eur / welfare_gamma_eur) * 100, 2)
  ) |>
  select(
    gamma, theta, T_star,
    welfare_gamma_eur,
    seller_profits_eur,
    buyer_profits_eur,
    seller_profit_share_percent,
    buyer_profit_share_percent
  ) |>
  arrange(theta, gamma)


profits_T_G <- welfare_with_profits_G_T |> 
  left_join(pg_welfare, by = c("gamma", "theta")) |> 
  mutate(
    buyer_profit_ratio = (buyer_profits_eur / buyer_profits_eur_pg) * 100
  )

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Plot Welfare Ratios

# Plot: Welfare Ratio (W_gamma / W_pg)
plot_profit_metric(
  data              = welfare_ratios,
  x_var             = "gamma",
  y_var             = "welfare_ratio",
  group_var         = "theta",
  color_var         = "theta",
  color_label       = expression(theta),
  colors            = theme_palette_theta,
  x_label           = expression(gamma),
  y_label           = "Welfare Ratios (%)",
  y_scale_percent   = TRUE,
  save_path         = with_T_fig_path,
  file_name         = "11_welfare_ratios_public_G_T.pdf",
  legend_position   = c(0.08, 0.1)  
)

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Plot Profits: Buyer and Seller

# Create the plot
plot_profit_metric(
  data              = welfare_with_profits_G_T,
  x_var             = "gamma",
  x_label           = expression(gamma),
  y_var             = "seller_profits_eur",
  group_var         = "theta",
  color_var         = "theta",
  colors            = theme_palette_theta,
  color_label       = expression(theta),
  y_label           = "Seller Profit (M€)",
  y_scale_million   = TRUE,
  save_path         = with_T_fig_path,
  file_name         = "12_seller_G_T_profit_vs_gamma_theta.pdf",
  legend_position   = c(0.08, 0.90)   
)


#  ---------------------------------------------------------------------- #
# Profits plot (Buyers)

# Create the plot
plot_profit_metric(
  data              = welfare_with_profits_G_T,
  x_var             = "gamma",
  x_label           = expression(gamma),
  y_var             = "buyer_profits_eur",
  group_var         = "theta",
  color_var         = "theta",
  colors            = theme_palette_theta,
  color_label       = expression(theta),
  y_label           = "Buyer Profit (M€)",
  y_scale_million   = TRUE,
  save_path         = with_T_fig_path,
  file_name         = "13_buyer_G_T_profit_vs_gamma_theta.pdf",
  legend_position   = c(0.08, 0.1)     
)


#  ---------------------------------------------------------------------- #
# Profits Share (%) - Sellers

# Create the plot
plot_profit_metric(
  data              = welfare_with_profits_G_T,
  x_var             = "gamma",
  x_label           = expression(gamma),
  y_var             = "seller_profit_share_percent",
  group_var         = "theta",
  color_var         = "theta",
  colors            = theme_palette_theta,
  color_label       = expression(theta),
  y_label           = "Seller Profit Share (%)",
  y_scale_percent   = TRUE,
  save_path         = with_T_fig_path,
  file_name         = "14_seller_G_T_profit_share_vs_gamma_theta.pdf",
  legend_position   = c(0.08, 0.90)     
)

#  ---------------------------------------------------------------------- #
# Profits Share (%) - Buyers

# Create the plot
plot_profit_metric(
  data              = welfare_with_profits_G_T,
  x_var             = "gamma",
  x_label           = expression(gamma),
  y_var             = "buyer_profit_share_percent",
  group_var         = "theta",
  color_var         = "theta",
  colors            = theme_palette_theta,
  color_label       = expression(theta),
  y_label           = "Buyer Profit Share (%)",
  y_scale_percent   = TRUE,
  save_path         = with_T_fig_path,
  file_name         = "15_buyer_G_T_profit_share_vs_gamma_theta.pdf",
  legend_position   = c(0.08, 0.1)     
)

#  ---------------------------------------------------------------------- #
# Profits Ratio - Buyer Profit (T=G) / Buyer Profit (P.G.) 

# Create the plot
plot_profit_metric(
  data              = profits_T_G,
  x_var             = "gamma",
  x_label           = expression(gamma),
  y_var             = "buyer_profit_ratio",
  group_var         = "theta",
  color_var         = "theta",
  color_label       = expression(theta),
  colors            = theme_palette_theta,
  y_label           = "Buyer Profit Ratio (%)",
  y_scale_percent   = TRUE,
  save_path         = with_T_fig_path,
  file_name         = "16_buyer_profit_ratio_G_T_vs_gamma_theta.pdf",
  legend_position   = c(0.08, 0.1)   
)


#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Save everything

# Define dataset list for comparison of T = G and Public Guarantees
datasets_T_G <- list(
  `Wind Solar Projects`        = results_with_T_G_comparison,
  `Equilibrium P. & Q.`        = equilibrium_prices_T,
  `Welfare with T star`        = welfare_dataset_G_T,
  `Welfare with Public G.`     = welfare_dataset_pg,
  `Welfare Ratios`             = welfare_ratios,
  `Profits when T=G`           = profits_T_G
)

# Filter all datasets by gamma
filtered_T_G <- lapply(datasets_T_G, filter_by_gamma)

# Construct dynamic filename
excel_filename_T_G <- "02_wind_solar_projects_cpr_public_expenditure_T_G.xlsx"
output_path_T_G <- file.path(with_T_path, excel_filename_T_G)

# Create workbook
wb_T_G <- createWorkbook()

# Add sheets dynamically
for (sheet in names(filtered_T_G)) {
  addWorksheet(wb_T_G, sheet)
  freezePane(wb_T_G, sheet, firstActiveRow = 2, firstActiveCol = 2)
  writeData(wb_T_G, sheet = sheet, x = filtered_T_G[[sheet]])
}

# Save workbook
saveWorkbook(wb_T_G, file = output_path_T_G, overwrite = TRUE)


# Section 8: Regulator-backed Contracts (RbC) -----------------------------

# Function to compute equilibrium prices for a given demand threshold variable (e.g., theta or theta_rbc)
compute_equilibrium_prices <- function(data, demand_var, expected_p, x) {

  demand_values <- unique(data[[demand_var]])
  
  map_dfr(demand_values, function(theta_val) {
    
    # 1. Get last project before reaching theta (for each gamma)
    before_theta <- data %>%
      filter(!!sym(demand_var) == theta_val) %>%
      group_by(gamma) %>%
      arrange(cumulative_capacity) %>%
      filter(cumulative_capacity < theta_val, xf_max_cpr < threshold_price) %>%
      slice_tail(n = 1) %>%
      ungroup()
    
    # 2. Get first project at or after theta
    after_theta <- data %>%
      filter(!!sym(demand_var) == theta_val) %>%
      group_by(gamma) %>%
      arrange(cumulative_capacity) %>%
      filter(cumulative_capacity >= theta_val) %>%
      slice_head(n = 1) %>%
      select(gamma, next_price = xf_max_cpr) %>%
      ungroup()
    
    # 3. Combine and compute equilibrium
    equilibrium <- before_theta %>%
      left_join(after_theta, by = "gamma") %>%
      mutate(
        equilibrium_price = if_else(
          !is.na(next_price) & next_price >= threshold_price,
          threshold_price,
          next_price
        ),
        equilibrium_quantity = cumulative_capacity,
        demand_label = demand_var,
        demand_value = theta_val
      ) %>%
      select(gamma, demand_label, demand_value, equilibrium_quantity, equilibrium_price)
    
    return(equilibrium)
  })
}

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

wind_solar_proj_2022_rbc_raw <- wind_solar_proj_2022 |> 
  crossing(gamma = gamma_values) |>
  rowwise() |> 
  mutate(
    # f_c_cpr --> find_f_root
    f_c_cpr = coalesce(
      find_f_root(q_i = q_i_mwh,
                  x = x,
                  gamma = gamma,
                  r_0 = r_0,
                  alpha = alpha,
                  beta = beta,
                  total_costs = total_cost,
                  T_values = 0),
      1),
    
    # f_spot_cpr : Pi_S - Pi_S0 = 0
    f_spot_cpr = coalesce(
      find_f_spot_root(q_i = q_i_mwh,
                       x = x,
                       gamma = gamma,
                       r_0 = r_0,
                       alpha = alpha,
                       beta = beta,
                       total_costs = total_cost,
                       expected_p = expected_p,
                       r = r,
                       T_values = 0),
      0),
    # f_upper_cpr : for \Pi_S that cross \Pi_S = 0 two times, find the second root.
    f_upper = 
      find_upper_root(q_i = q_i_mwh,
                      x = x,
                      gamma = gamma,
                      r_0 = r_0,
                      alpha = alpha,
                      beta = beta,
                      total_costs = total_cost,
                      T_values = 0)
  ) |> 
  ungroup() |> 
  mutate(f_upper_message = if_else(
    !is.na(f_upper),
    if_else(f_upper > expected_p, "f_upper > E(p)", "f_upper <= E(p)"),
    NA_character_
  ),
  f_max_cpr = pmax(f_c_cpr, f_spot_cpr, na.rm = TRUE),
  xf_c_cpr = x * f_c_cpr,
  xf_spot_cpr = x * f_spot_cpr,
  xf_upper_cpr = x * f_upper,
  xf_max_cpr = x * f_max_cpr
  ) 


wind_solar_proj_2022_rbc <- wind_solar_proj_2022_rbc_raw |> 
  arrange(gamma, xf_max_cpr) |> 
  group_by(gamma) |> 
  mutate(cumulative_production = cumsum(q_i_mwh),
         cumulative_capacity = cumsum(capacity),
         x_q_exp_p_total_costs = x * expected_p * q_i_mwh - total_cost
  ) |>
  ungroup()


wind_solar_proj_2022_rbc <- wind_solar_proj_2022_rbc |> 
  mutate(theta = total_contract_demand,
         theta_private = private_demand,
         theta_rbc = rbc_demand
         ) |> 
  arrange(theta, xf_max_cpr, gamma)

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

## Equilibrium Prices Dataset Creation -----------------------------------

equilibrium_theta <- compute_equilibrium_prices(wind_solar_proj_2022_rbc, "theta", expected_p, x)
equilibrium_theta_private <- compute_equilibrium_prices(wind_solar_proj_2022_rbc, "theta_private", expected_p, x)

# Combine if needed
all_equilibria <- bind_rows(equilibrium_theta, equilibrium_theta_private)

all_equilibria <- all_equilibria %>%
  arrange(demand_label, demand_value, gamma)

# View results
print(all_equilibria)

equilibrium_prices <- all_equilibria %>% 
  mutate(theta = demand_value) %>% 
  filter(demand_label == "theta") %>% 
  select(-demand_value)

#  ---------------------------------------------------------------------- #

## Welfare Dataset Creation ----------------------------------------------

# Join theta = 5000 and respective equilibria prices and quantities with main 
# dataset.
theta_equilibrium <- all_equilibria %>%
  filter(demand_label == "theta") %>%
  select(
    gamma,
    theta = demand_value,
    theta_equilibrium_price = equilibrium_price,
    theta_equilibrium_quantity = equilibrium_quantity
  )


wind_solar_proj_2022_rbc <- wind_solar_proj_2022_rbc %>%
  left_join(theta_equilibrium, by = c("gamma", "theta"))


wind_solar_proj_2022_rbc <- wind_solar_proj_2022_rbc |> 
  mutate(f_equilibrium = theta_equilibrium_price / x,
         xf_equilibrium = theta_equilibrium_price,
         equilibrium_quantity = theta_equilibrium_quantity
         ) |> 
  select(-theta_equilibrium_price, -theta_equilibrium_quantity)


wind_solar_proj_2022_rbc <- wind_solar_proj_2022_rbc |>
  group_by(gamma) |> 
  rowwise() |> 
  mutate(R_f_equilibrium_cpr = 
           compute_R_value_gamma(f_equilibrium, gamma, q_i_mwh, x, r_0, alpha, beta)) |> 
  ungroup() |> 
  mutate(
    x_q_exp_p_total_costs_R = x * expected_p * q_i_mwh - total_cost
    - R_f_equilibrium_cpr
  )

wind_solar_proj_2022_rbc <- wind_solar_proj_2022_rbc |> 
  mutate(
    profit_cpr_private_contracts = 
      vectorized_pi_s(
        f = f_equilibrium,
        q_i = q_i_mwh,
        x = x,
        gamma = gamma,
        r_0 = r_0,
        alpha = alpha,
        beta = beta,
        total_costs = total_cost,
        T_values = 0
      )
  ) |> 
  arrange(theta, gamma, xf_max_cpr)



ordered_vars <- c(
  "theta", "gamma", "f_c_cpr", "f_spot_cpr", "f_upper", "f_upper_message",
  "f_max_cpr", "f_equilibrium", "R_f_equilibrium_cpr",
  "xf_c_cpr", "xf_spot_cpr", "xf_upper_cpr", "xf_max_cpr", "xf_equilibrium",
  "equilibrium_quantity", "x_q_exp_p_total_costs", "x_q_exp_p_total_costs_R",
  "cumulative_production", "cumulative_capacity", "profit_cpr_private_contracts"
)

# Reorder dataframe
wind_solar_proj_2022_rbc <- wind_solar_proj_2022_rbc |> 
  select(
    everything(),              # Keeps all variables in current order...
    -all_of(ordered_vars),     # ...but temporarily removes the ones you want to reorder
    -profits_sp_no_cpr,                # temporarily remove chosen_xf to place it explicitly
    profits_sp_no_cpr,                 # put chosen_xf in place
    all_of(ordered_vars)       # then add ordered vars in the desired order
  )

# STEP 1: Compute W^0 for gamma == 0
W_0_df <- wind_solar_proj_2022_rbc |> 
  filter(gamma == 0, profits_sp_no_cpr >= 0) |> 
  group_by(theta) |> 
  summarise(W_0 = sum(x_q_exp_p_total_costs - r, na.rm = TRUE), .groups = "drop")

# STEP 2: Compute W(gamma) for each gamma
W_gamma_df <- wind_solar_proj_2022_rbc |> 
  group_by(gamma, theta) |> 
  summarise(
    welfare_gamma_eur = sum(
      if_else(xf_max_cpr <= xf_equilibrium & cumulative_capacity < equilibrium_quantity,
              x_q_exp_p_total_costs_R, 0),
      na.rm = TRUE
    )
  ) |> 
  ungroup()

# STEP 3: Compute W(gamma) for each gamma == 0
W_gamma_0_df <- W_gamma_df %>%
  filter(gamma == 0) %>%
  rename(welfare_0_eur = welfare_gamma_eur) %>%
  select(theta, welfare_0_eur)


# STEP 4: Equilibrium prices for gamma == 0
eq_gamma_0 <- equilibrium_prices %>%
  filter(gamma == 0) %>%
  select(theta, 
         equilibrium_quantity_gamma_0 = equilibrium_quantity,
         equilibrium_price_gamma_0 = equilibrium_price)


# STEP 5: Combine all metrics into one final table

welfare_dataset_wo_rbc_5000 <- equilibrium_prices |> 
  left_join(W_gamma_df, by = c("gamma", "theta")) |> 
  left_join(W_gamma_0_df, by = "theta") |> 
  left_join(W_0_df, by = "theta") |> 
  left_join(eq_gamma_0, by = "theta") |> 
  mutate(
    eq_price = equilibrium_price,
    eq_quantity = equilibrium_quantity,
    
    welfare_ratio_percent      = (welfare_gamma_eur / welfare_0_eur) * 100,
    welfare_gap_million_eur    = (welfare_0_eur - welfare_gamma_eur) / 1e6,
    welfare_gain_million_eur   = (welfare_gamma_eur - W_0) / 1e6,
    
    eq_quantity_ratio_percent  = (eq_quantity / equilibrium_quantity_gamma_0) * 100,
    eq_price_ratio_percent     = (eq_price / equilibrium_price_gamma_0) * 100,
    
    W_0 = if_else(gamma == 0, W_0, NA_real_)  # Keep only at gamma = 0 if needed
  ) |> 
  select(
    gamma, theta,
    eq_price, eq_price_ratio_percent,
    eq_quantity, eq_quantity_ratio_percent,
    W_0, welfare_gamma_eur, welfare_ratio_percent,
    welfare_gap_million_eur, welfare_gain_million_eur
  ) |>
  arrange(theta, gamma)

welfare_dataset_wo_rbc_5000

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

## Profits ---------------------------------------------------------------

# Profits by gamma and theta

profits_by_gamma_theta_wo_rbc_5000 <- wind_solar_proj_2022_rbc |> 
  group_by(gamma, theta) |> 
  filter(xf_max_cpr <= xf_equilibrium & cumulative_capacity < equilibrium_quantity) |> 
  summarise(
    seller_profits_eur  = sum(profit_cpr_private_contracts, na.rm = TRUE),
    .groups = "drop"
  ) |> 
  ungroup() |> 
  arrange(theta, gamma, seller_profits_eur)

profits_by_gamma_theta_wo_rbc_5000 <- profits_by_gamma_theta_wo_rbc_5000 |> 
  left_join(welfare_dataset_wo_rbc_5000 |> 
              select(gamma, theta, welfare_gamma_eur),
            by = c("gamma", "theta")) |> 
  mutate(buyer_profits_eur = welfare_gamma_eur - seller_profits_eur) |> 
  select(-welfare_gamma_eur)

welfare_with_profits <- welfare_dataset_wo_rbc_5000 %>%
  left_join(profits_by_gamma_theta_wo_rbc_5000, by = c("gamma", "theta"))


welfare_with_profits <- welfare_with_profits %>%
  mutate(
    seller_profit_share_percent = round(
      (seller_profits_eur / welfare_gamma_eur) * 100, 2),
    buyer_profit_share_percent = round(
      (buyer_profits_eur / welfare_gamma_eur) * 100, 2),
  ) |> 
  select(gamma, theta, seller_profits_eur,
         buyer_profits_eur, seller_profit_share_percent, buyer_profit_share_percent, welfare_gamma_eur) |> 
  arrange(theta, gamma)

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Save this in a separate workbook, for clarity
# Save everything

# Extract theta value
theta_value <- wind_solar_proj_2022_rbc$theta[1]

# Construct dynamic filename and output path
excel_filename_rbc <- paste0("01_wind_solar_projects_wo_rbc_theta_", theta_value, ".xlsx")
output_path_rbc <- file.path(with_rbc_path, excel_filename_rbc)

# Define dataset list
datasets_rbc <- list(
  `Wind_Solar_Projects`     = wind_solar_proj_2022_rbc,
  `Equilibrium P. & Q.`     = theta_equilibrium,
  `Welfare`                 = welfare_dataset_wo_rbc_5000,
  `Profits`                 = welfare_with_profits
)

# Filter all datasets by selected gamma values
filtered_rbc <- lapply(datasets_rbc, filter_by_gamma)

# Create workbook
wb_rbc <- createWorkbook()

# Add sheets dynamically
for (sheet in names(filtered_rbc)) {
  addWorksheet(wb_rbc, sheet)
  freezePane(wb_rbc, sheet, firstActiveRow = 2, firstActiveCol = 2)
  writeData(wb_rbc, sheet = sheet, x = filtered_rbc[[sheet]])
}

# Save workbook
saveWorkbook(wb_rbc, file = output_path_rbc, overwrite = TRUE)


#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

## f^R Computation --------------------------------------------------------

# First we compute the f_rbc for each plant. 

wind_solar_proj_2022_rbc <- wind_solar_proj_2022_rbc %>%
  mutate(
    winning_plant = xf_max_cpr <= xf_equilibrium & cumulative_capacity <= equilibrium_quantity
  ) %>%
  filter(winning_plant) |> 
  select(-winning_plant) |> 
  mutate(
    f_rbc = (profit_cpr_private_contracts + total_cost) / (q_i_mwh * x),
    xf_rbc = x * f_rbc,
    # Profits should be the same as in private market
    # Let us compute a variable to see if they are the same. Yes, they are.
    profits_check_f_rbc_equals_private = xf_rbc * q_i_mwh - total_cost
  ) %>%
  ungroup() %>% 
  select(-profits_check_f_rbc_equals_private)


wind_solar_proj_2022_rbc <- wind_solar_proj_2022_rbc |> 
  arrange(gamma, f_rbc) |> 
  group_by(gamma) |> 
  mutate(cumulative_production = cumsum(q_i_mwh),
         cumulative_capacity = cumsum(capacity)
  ) |>
  ungroup()


### P & Q for theta RBC --------------------------------------------------
# Be careful, those are not the equilibrium prices and quantitities per se.
# We just want to figure out which ones have private contracts, and each ones
# have regulator-backed contracts with theta_rbc.
# We subsequently compute the equilibrium prices/quantities.

# 1. For each gamma, get the last row before theta_val (preserving cumulative capacity)
before_theta_rbc <- wind_solar_proj_2022_rbc %>%
  group_by(gamma) %>%
  arrange(cumulative_capacity) %>%
  filter(cumulative_capacity < theta_rbc, 
         xf_rbc < threshold_price) %>%
  slice_tail(n = 1) %>%
  ungroup()

# 2. For each gamma, get the first row at or after theta_val (to peek at the next price)
after_theta_rbc <- wind_solar_proj_2022_rbc %>%
  group_by(gamma) %>%
  arrange(cumulative_capacity) %>%
  filter(cumulative_capacity >= theta_rbc) %>%
  slice_head(n = 1) %>%
  select(gamma, next_price = xf_rbc) %>%
  ungroup()

# 3. Join the two and update the price:
#    - Keep the cumulative capacity from before_theta
#    - Update the price to threshold_price only if the next row's price is >= threshold_price.
equilibrium_prices <- before_theta_rbc %>%
  left_join(after_theta_rbc, by = "gamma") |> 
  mutate(
    equilibrium_price = if_else(
      !is.na(next_price) & next_price >= threshold_price,
      threshold_price,
      next_price
    ),
    equilibrium_quantity = cumulative_capacity
  ) %>%
  select(gamma, equilibrium_quantity, equilibrium_price, theta, theta_rbc)



# View the combined results
print(equilibrium_prices)

equilibrium_prices <- equilibrium_prices |> 
  arrange(theta, gamma) %>% 
  rename(xf_rbc_equilibrium = equilibrium_price,
         quantity_rbc = equilibrium_quantity) |> 
  mutate(f_rbc_equilibrium = xf_rbc_equilibrium / x)

# Get all unique theta values
theta_values <- sort(unique(equilibrium_prices$theta_rbc))

wind_solar_proj_2022_rbc <- wind_solar_proj_2022_rbc |> 
  left_join(
    equilibrium_prices |> 
      select(gamma, theta_rbc, xf_rbc_equilibrium, f_rbc_equilibrium, quantity_rbc),
    by = c("gamma", "theta_rbc")
  )

#  ---------------------------------------------------------------------- #

### Profits for f_R^* ----------------------------------------------------

# Now we define the f* and q* for each gamma. It is the marginal just below
# the RBC demand 2500.

marginal_f_rbc <- wind_solar_proj_2022_rbc %>%
  group_by(gamma, theta_rbc) %>%
  arrange(cumulative_capacity, .by_group = TRUE) %>%  # Important: sort within each gamma/theta_rbc
  filter(cumulative_capacity == quantity_rbc) %>%
  ungroup() %>%
  select(gamma, theta_rbc, f_rbc_equilibrium, xf_rbc_equilibrium)

#
marginal_flags <- wind_solar_proj_2022_rbc %>%
  group_by(gamma, theta_rbc) %>%
  mutate(
    f_rbc_equilibrium = (profit_cpr_private_contracts + total_cost) / (x * q_i_mwh),
    xf_rbc_equilibrium = x * f_rbc_equilibrium
  ) %>%
  group_by(gamma) %>%
  filter(
    near(xf_rbc_equilibrium, xf_rbc),
    near(cumulative_capacity, quantity_rbc)
  ) %>%
  slice(1) %>%
  mutate(is_marginal = TRUE) %>%
  ungroup()

# Profits computation for RBC cases
wind_solar_proj_2022_rbc <- wind_solar_proj_2022_rbc %>%
  mutate(
    profit_f_rbc = (xf_rbc_equilibrium * q_i_mwh) - total_cost
  ) %>%
  relocate(f_rbc_equilibrium, .before = xf_rbc_equilibrium)


wind_solar_proj_2022_rbc <- wind_solar_proj_2022_rbc %>%
  left_join(
    select(marginal_flags, gamma, theta_rbc, cumulative_capacity, is_marginal),
    by = c("gamma", "theta_rbc", "cumulative_capacity")
  ) %>%
  mutate(
    is_marginal = if_else(is.na(is_marginal), FALSE, is_marginal),
    cumulative_cutoff_rbc = cumulative_capacity <= quantity_rbc & xf_rbc <= xf_rbc_equilibrium
  )

wind_solar_proj_2022_rbc <- wind_solar_proj_2022_rbc %>%
  mutate(
    profits = case_when(
      xf_max_cpr <= xf_equilibrium & cumulative_cutoff_rbc ~ profit_f_rbc,
      xf_equilibrium & xf_rbc >= xf_rbc_equilibrium & cumulative_capacity <= theta ~ profit_cpr_private_contracts,
      TRUE ~ NA
    ),
    contract_type = case_when(
      xf_max_cpr <= xf_equilibrium & cumulative_cutoff_rbc ~ "Regulated",
      xf_max_cpr <= xf_equilibrium & xf_rbc >= xf_rbc_equilibrium & cumulative_capacity <= theta ~ "Private",
      TRUE ~ "No Private/Regulated"
    )
  ) |> 
  arrange(theta_rbc, gamma, xf_rbc) |> 
  select(-is_marginal, -cumulative_cutoff_rbc)

# Calculate cumulative sums for private contract only for further use
private_cumsum <- wind_solar_proj_2022_rbc %>%
  filter(contract_type == "Private") %>%
  arrange(theta_rbc, gamma, xf_rbc) %>%
  group_by(gamma, theta_rbc) %>%
  mutate(
    cumulative_capacity_private = cumsum(capacity),
    cumulative_production_private = cumsum(q_i_mwh)
  ) %>%
  ungroup() |> 
  group_by(theta_rbc, gamma) |> 
  summarise(eq_quantity_private = last(cumulative_capacity_private), .groups = "drop") |> 
  ungroup() |> 
  select(-theta_rbc)

# We should have that profits under Regulated scheme should be >= profits under Private scheme
# some floating errors, no worries to have. Everything is fine!
violations <- wind_solar_proj_2022_rbc %>%
  filter(
    contract_type == "Regulated",
    profit_f_rbc < profit_cpr_private_contracts
  )

#  ---------------------------------------------------------------------- #

### Welfare --------------------------------------------------------------


# Create a table with eq_p and eq_q for each gamma

# Compute W^0 for gamma == 0
W_0_df <- wind_solar_proj_2022_rbc |> 
  filter(gamma == 0, profits_sp_no_cpr >= 0) |> 
  group_by(theta) |> 
  summarise(theta_rbc = first(theta_rbc),
            W_0 = sum(x_q_exp_p_total_costs - r, na.rm = TRUE), .groups = "drop",
            )

# STEP 4: Compute W(gamma) for each gamma. We decide to do the sum for both
# Private and Regulated contracts. We also can just take the 
# sum(x_q_exp_p_total_costs_R) for only "Private" contracts for 
# W_gamma_private_c and the sum(x_q_exp_p_total_costs) for the W_gamma_rbc
# variable

W_gamma_private_c <- wind_solar_proj_2022_rbc %>%
  filter(contract_type == "Private") %>%
  group_by(gamma) %>%
  summarise(
    theta = first(theta),
    theta_rbc = first(theta_rbc),
    W_private_c = sum(x_q_exp_p_total_costs_R, na.rm = TRUE),
    .groups = "drop"
  )

W_gamma_rbc <- wind_solar_proj_2022_rbc %>%
  filter(
    contract_type == "Regulated"
  ) %>%
  group_by(gamma) %>%
  summarise(
    theta = first(theta),
    theta_rbc = first(theta_rbc), 
    # We could also create W_rbc = sum(R_f_equilibrium_cpr), na.rm = TRUE),
    # .groups = "drop"
    # But if we do this, we should specify above in W_gamma_private_c:
    # filter(contract_type %in% c("Private", "Regulated"))
    W_rbc = sum(x_q_exp_p_total_costs, na.rm = TRUE),
    .groups = "drop"
  )

W_gamma_df <- W_gamma_private_c %>% 
  left_join(W_gamma_rbc, 
            by = c("gamma", "theta", "theta_rbc")) %>% 
  mutate(
    W_gamma = W_private_c + W_rbc
  ) %>%
  arrange(gamma, theta)


W_gamma_0_df <- W_gamma_df %>%
  filter(gamma == 0) %>%
  rename(W_gamma_0 = W_gamma) %>%
  select(theta, theta_rbc, W_gamma_0)


eq_gamma_0 <- equilibrium_prices %>%
  filter(gamma == 0) %>%
  select(theta, theta_rbc,
         quantity_gamma_0 = quantity_rbc,
         price_gamma_0 = xf_rbc_equilibrium
  )

equilibrium_prices

# STEP 5: Combine all metrics into one final table

welfare_dataset_rbc <- equilibrium_prices |> 
  left_join(W_gamma_df |> select(-theta), by = c("gamma", "theta_rbc")) |> 
  left_join(W_gamma_0_df |> select(-theta), by = "theta_rbc") |> 
  left_join(W_0_df |> select(-theta), by = "theta_rbc") |> 
  left_join(eq_gamma_0 |> select(-theta), by = "theta_rbc") |> 
  mutate(
    eq_quantity_rbc = quantity_rbc,
    eq_price_rbc = xf_rbc_equilibrium,
    
    welfare_0_eur = if_else(gamma == 0, W_0, NA_real_),
    welfare_gamma_eur = W_gamma,
    welfare_gamma_ratio_percent = (W_gamma / W_gamma_0) * 100,
    welfare_gap_meur = (W_gamma_0 - W_gamma) / 1e6,
    welfare_gain_meur = (W_gamma - W_0) / 1e6
  ) |> 
  
  select(
    gamma, theta_rbc, eq_price_rbc, eq_quantity_rbc,
    welfare_0_eur, welfare_gamma_eur, welfare_gamma_ratio_percent,
    welfare_gap_meur, welfare_gain_meur
  ) |> 
  
  arrange(theta_rbc, gamma)


# For further use, we will merge dataset with theta = 5000 w/o RBC
# and also add the cumulative of private combtract
welfare_dataset_rbc <- welfare_dataset_rbc %>%
  left_join(
    welfare_dataset_wo_rbc_5000 %>%
      select(gamma, theta, eq_price_wo_rbc_5000 = eq_price, eq_quantity_total = eq_quantity),  # rename for clarity
    by = "gamma"
  ) |> 
  left_join(private_cumsum |>
              select(gamma, eq_quantity_private), by = "gamma"
  ) |> 
  select(-theta) |> 
  relocate(eq_price_wo_rbc_5000, .after = eq_price_rbc) |> 
  relocate(eq_quantity_private, .after = eq_quantity_rbc) |> 
  relocate(eq_quantity_total, .after = eq_quantity_private) |> 
  mutate(eq_price_rbc_vs_no_rbc_ratio_percent = (eq_price_rbc / eq_price_wo_rbc_5000) * 100,
         eq_quantity_rbc_vs_no_rbc_ratio_percent = (eq_quantity_rbc / eq_quantity_private) * 100) |> 
  relocate(eq_price_rbc_vs_no_rbc_ratio_percent, .after = eq_price_wo_rbc_5000) |> 
  relocate(eq_quantity_rbc_vs_no_rbc_ratio_percent, .after = eq_quantity_total) 


#  ---------------------------------------------------------------------- #

### Profits --------------------------------------------------------------

# Profits by gamma and theta

# Firms with private contracts
profits_rbc <- wind_solar_proj_2022_rbc %>%
  filter(contract_type %in% c("Private", "Regulated")) %>%
  group_by(gamma) %>%
  summarise(
    theta_rbc = first(theta_rbc),
    profits_private = sum(
      if_else(contract_type == "Private", profits, 0),
      na.rm = TRUE
    ),
    profits_rbc = sum(
      if_else(contract_type == "Regulated", profits, 0),
      na.rm = TRUE
    ),
    seller_profits_eur_rbc = sum(profits, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(gamma)

# Capture the demand value and the theta_rbc value you want to use

theta_equilibrium <- unique(equilibrium_theta$demand_value)  
theta_rbc_value <- unique(profits_rbc$theta_rbc)             

# Step 1: Merge welfare_dataset_rbc (for RBC equilibrium info)
profits_rbc <- profits_rbc %>%
  left_join(
    welfare_dataset_rbc %>%
      select(
        gamma,
        theta_rbc,
        welfare_gamma_eur,
        eq_price_rbc,
        eq_quantity_rbc
      ),
    by = c("gamma", "theta_rbc")
  ) %>%
  mutate(
    buyer_profits_eur_rbc = welfare_gamma_eur - seller_profits_eur_rbc
  ) %>%
  relocate(buyer_profits_eur_rbc, .after = seller_profits_eur_rbc)

# Dynamically rename the RBC columns
profits_rbc <- profits_rbc %>%
  rename_with(
    .fn = ~glue("Eq. Price (RBC, theta = {theta_rbc_value})"),
    .cols = eq_price_rbc
  ) %>%
  rename_with(
    .fn = ~glue("Eq. Quantity (RBC, theta = {theta_rbc_value})"),
    .cols = eq_quantity_rbc
  )

# Step 2: Merge equilibrium_theta (external equilibrium info)
profits_rbc <- profits_rbc %>%
  left_join(
    equilibrium_theta %>%
      filter(demand_value == theta_equilibrium) %>%
      select(gamma,
             equilibrium_quantity,
             equilibrium_price),
    by = "gamma"
  ) %>%
  rename_with(
    ~glue("Eq. Quantity (theta = {theta_equilibrium})"),
    .cols = equilibrium_quantity
  ) %>%
  rename_with(
    ~glue("Eq. Price (theta = {theta_equilibrium})"),
    .cols = equilibrium_price
  )

profits_rbc <- profits_rbc %>%
  relocate(starts_with("Eq."), .after = gamma) 

wind_solar_proj_2022_rbc <- wind_solar_proj_2022_rbc %>%
  relocate(profit_cpr_private_contracts, .before = profit_f_rbc) %>%
  relocate(f_rbc_equilibrium, xf_rbc_equilibrium, .after = cumulative_capacity) %>%
  relocate(xf_rbc_equilibrium, quantity_rbc, .after = xf_rbc) %>%
  arrange(theta, gamma, xf_rbc)


### Welfare Comparison----------------------------------------------------

# Now we will create some tables that allows to compare the RBC with other
# cases, please. We will do that for theta = 2,500 and compare RBC with:

# i.   Public Subsidies
# ii.  Public Guarantees when T = G
# iii. Baseline Results

# We will compare welfare and seller/buyer profits

# Public Guarantees when T = G
welfare_with_profits_G_T <- welfare_with_profits_G_T |> 
  filter(theta == theta_comparison_rbc) |> 
  rename(welfare_gamma_eur_g_t  = welfare_gamma_eur,
         seller_profits_eur_g_t = seller_profits_eur,
         buyer_profits_eur_g_t  = buyer_profits_eur
  ) |> 
  select(-seller_profit_share_percent, -buyer_profit_share_percent)

# Baseline
welfare_with_profits_baseline <- welfare_with_profits_baseline |> 
  filter(theta == theta_comparison_rbc) |> 
  rename(welfare_gamma_eur_b  = welfare_gamma_eur,
         seller_profits_eur_b = seller_profits_eur,
         buyer_profits_eur_b  = buyer_profits_eur
  ) |> 
  select(-seller_profit_share_percent, -buyer_profit_share_percent)


welfare_dataset_pg <- welfare_dataset_pg |> 
  filter(theta == theta_comparison_rbc) |> 
  rename(welfare_gamma_eur_pg   = welfare_gamma_eur
  ) |> 
  select(gamma, theta, buyer_profits_eur_pg, seller_profits_eur_pg, welfare_gamma_eur_pg)

profits_rbc <- profits_rbc |> 
  rename(theta = theta_rbc,
         welfare_gamma_eur_rbc = welfare_gamma_eur)

combined_profits_welfare <- profits_rbc |> 
  left_join(welfare_with_profits_G_T, by = c("gamma", "theta")) |> 
  left_join(welfare_with_profits_baseline, by = c("gamma", "theta")) |> 
  left_join(welfare_dataset_pg, by = c("gamma", "theta")) |> 
  select(-T_star) |> 
  relocate(theta, .before = gamma)

combined_profits_welfare_ratios <- combined_profits_welfare |> 
  mutate(
    # Welfare ratios
    welfare_ratio_rbc_vs_g_t      = welfare_gamma_eur_rbc / welfare_gamma_eur_g_t,
    welfare_ratio_rbc_vs_baseline = welfare_gamma_eur_rbc / welfare_gamma_eur_b,
    welfare_ratio_rbc_vs_pg       = welfare_gamma_eur_rbc / welfare_gamma_eur_pg,
    
    # Seller profit ratios
    seller_profit_ratio_rbc_vs_g_t      = seller_profits_eur_rbc / seller_profits_eur_g_t,
    seller_profit_ratio_rbc_vs_baseline = seller_profits_eur_rbc / seller_profits_eur_b,
    seller_profit_ratio_rbc_vs_pg       = seller_profits_eur_rbc / seller_profits_eur_pg,
    
    # Buyer profit ratios
    buyer_profit_ratio_rbc_vs_g_t      = buyer_profits_eur_rbc / buyer_profits_eur_g_t,
    buyer_profit_ratio_rbc_vs_baseline = buyer_profits_eur_rbc / buyer_profits_eur_b,
    buyer_profit_ratio_rbc_vs_pg       = buyer_profits_eur_rbc / buyer_profits_eur_pg
  )

# Useful for plotting: from wide to long

combined_profits_ratios_long <- combined_profits_welfare_ratios |>
  select(gamma,
         welfare_ratio_rbc_vs_pg,
         welfare_ratio_rbc_vs_g_t,
         welfare_ratio_rbc_vs_baseline,
         seller_profit_ratio_rbc_vs_pg,
         seller_profit_ratio_rbc_vs_g_t,
         seller_profit_ratio_rbc_vs_baseline,
         buyer_profit_ratio_rbc_vs_pg,
         buyer_profit_ratio_rbc_vs_g_t,
         buyer_profit_ratio_rbc_vs_baseline) |> 
  pivot_longer(
    cols = -gamma,
    names_to = c("type", "scenario"),
    names_pattern = "(welfare_ratio|seller_profit_ratio|buyer_profit_ratio)_rbc_vs_(pg|g_t|baseline)",
    values_to = "value"
  )

#### Plots ---------------------------------------------------------------

plot_ratio_by_type <- function(
    data_long,
    ratio_type,
    base_size = 25,
    filename = NULL,
    folder = NULL,
    base_palette = theme_palette_welfare,
    custom_labels = NULL
) {
  # Filter data
  df <- data_long |>
    dplyr::filter(type == ratio_type)
  
  # Handle custom label alignment
  if (!is.null(custom_labels)) {
    scenario_levels <- names(custom_labels)
    df <- df |>
      dplyr::mutate(scenario = factor(scenario, levels = scenario_levels))
    scenarios <- scenario_levels
    show_legend <- TRUE
    labels <- custom_labels
  } else {
    df <- df |>
      dplyr::mutate(scenario = as.factor(scenario))
    scenarios <- levels(df$scenario)
    show_legend <- FALSE
    labels <- scenarios
  }
  
  num_scenarios <- length(scenarios)
  
  # Aesthetic mappings
  color_values <- scales::gradient_n_pal(base_palette)(seq(0, 1, length.out = num_scenarios))
  names(color_values) <- scenarios
  
  linetypes <- setNames(rep(c("solid", "dashed", "dotted", "dotdash", "twodash"), length.out = num_scenarios), scenarios)
  shapes <- setNames(rep(16:20, length.out = num_scenarios), scenarios) # Solid shapes
  
  # Dynamic legend positioning
  gamma_cutoff <- quantile(df$gamma, 0.1, na.rm = TRUE)
  value_cutoff <- quantile(df$value, 0.5, na.rm = TRUE)
  
  top_left_density <- df |> filter(gamma <= gamma_cutoff, value >= value_cutoff) |> nrow()
  bottom_left_density <- df |> filter(gamma <= gamma_cutoff, value < value_cutoff) |> nrow()
  
  legend_at_top <- top_left_density < bottom_left_density
  legend_position <- if (legend_at_top) c(0.05, 0.95) else c(0.05, 0.05)
  legend_just <- if (legend_at_top) c("left", "top") else c("left", "bottom")
  
  # Build plot
  p <- ggplot(df, ggplot2::aes(
    x = gamma,
    y = value,
    color = scenario,
    linetype = scenario,
    shape = scenario
  )) +
    geom_line(linewidth = 1) +
    geom_point(size = 3) +
    scale_color_manual(values = color_values, labels = labels, name = NULL) +
    scale_linetype_manual(values = linetypes, labels = labels, name = NULL) +
    scale_shape_manual(values = shapes, labels = labels, name = NULL) +
    scale_y_continuous(labels = scales::label_number(accuracy = 0.01)) +
    labs(
      x = expression(gamma),
      y = switch(
        ratio_type,
        "welfare_ratio" = "Welfare Ratio",
        "seller_profit_ratio" = "Seller Profit Ratio",
        "buyer_profit_ratio" = "Buyer Profit Ratio",
        ratio_type
      )
    ) +
    theme_minimal(base_size = base_size) +
    theme(
      legend.position = if (show_legend) legend_position else "none",
      legend.justification = legend_just,
      legend.box.just = "left",
      legend.background = element_rect(fill = scales::alpha("white", 0.1), color = NA),
      legend.margin = margin(4, 4, 4, 4),
      legend.text = element_text(size = base_size * 0.6),
      legend.key.size = unit(0.4, "cm"),
      panel.grid.major = element_line(color = "grey90", size = 0.2),
      panel.grid.minor = element_line(color = "grey95", size = 0.1)
    )
  
  # Legend override
  if (show_legend) {
    p <- p + guides(
      color = guide_legend(
        override.aes = list(
          linetype = unname(linetypes),
          shape = unname(shapes),
          color = unname(color_values)
        )
      )
    )
  }
  
  # Save plot
  if (!is.null(filename) && !is.null(folder)) {
    path <- file.path(folder, filename)
    ggsave(path, plot = p, width = 16, height = 9, dpi = 300)
    message("✅ Saved plot to: ", path)
  }
  
  print(p)
  invisible(p)
}



# Calls of functions
plot_ratio_by_type(
  combined_profits_ratios_long,
  ratio_type = "welfare_ratio",
  filename = "01_welfare_rbc_vs_others.pdf",
  folder = with_rbc_fig_path,
  custom_labels = c(
    pg       = expression(frac(W[RBC], W[PG])),
    g_t      = expression(frac(W[RBC], W["G=T"])),
    baseline = expression(frac(W[RBC], W[Baseline]))
  )
)


plot_ratio_by_type(
  combined_profits_ratios_long,
  ratio_type = "seller_profit_ratio",
  filename = "02_seller_profit_rbc_vs_others.pdf",
  folder = with_rbc_fig_path,
  custom_labels = c(
    pg       = expression(frac(Pi[RBC]^S, Pi[PG]^S)),
    g_t      = expression(frac(Pi[RBC]^S, Pi["G=T"]^S)),
    baseline = expression(frac(Pi[RBC]^S, Pi[Baseline]^S))
  )
)

plot_ratio_by_type(
  combined_profits_ratios_long,
  ratio_type = "buyer_profit_ratio",
  filename = "03_buyer_profit_rbc_vs_others.pdf",
  folder = with_rbc_fig_path,
  custom_labels = c(
    pg       = expression(frac(Pi[RBC]^B, Pi[PG]^B)),
    g_t      = expression(frac(Pi[RBC]^B, Pi["G=T"]^B)),
    baseline = expression(frac(Pi[RBC]^B, Pi[Baseline]^B))
  )
)



#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

## Welfare w/o RBC (2500) ------------------------------------------------

wind_solar_proj_2022_wo_rbc_2500 <- wind_solar_proj_2022_rbc_raw |> 
  arrange(gamma, xf_max_cpr) |> 
  group_by(gamma) |> 
  mutate(cumulative_production = cumsum(q_i_mwh),
         cumulative_capacity = cumsum(capacity),
         x_q_exp_p_total_costs = x * expected_p * q_i_mwh - total_cost
  ) |>
  ungroup()


wind_solar_proj_2022_wo_rbc_2500 <- wind_solar_proj_2022_wo_rbc_2500 |> 
  mutate(theta = total_contract_demand,
         theta_private = private_demand,
         theta_rbc = rbc_demand
  ) |> 
  arrange(theta, xf_max_cpr, gamma)

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

#  ---------------------------------------------------------------------- #

## Equilibrium Prices Dataset Creation -----------------------------------

equilibrium_theta_private <- compute_equilibrium_prices(wind_solar_proj_2022_wo_rbc_2500, "theta_private", expected_p, x)

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

## Welfare Dataset Creation ----------------------------------------------

# Join theta_private = 2500 and respective equilibria prices and quantities with main 
# dataset.
theta_private_equilibrium <- equilibrium_theta_private %>%
  filter(demand_label == "theta_private") %>%
  select(
    gamma,
    theta_private = demand_value,
    private_equilibrium_price = equilibrium_price,
    private_equilibrium_quantity = equilibrium_quantity
  )


wind_solar_proj_2022_wo_rbc_2500 <- wind_solar_proj_2022_wo_rbc_2500 %>%
  left_join(theta_private_equilibrium, by = c("gamma", "theta_private"))


wind_solar_proj_2022_wo_rbc_2500 <- wind_solar_proj_2022_wo_rbc_2500 |> 
  mutate(f_private_equilibrium = private_equilibrium_price / x,
         xf_private_equilibrium = private_equilibrium_price
  ) |> 
  select(-private_equilibrium_price)


wind_solar_proj_2022_wo_rbc_2500 <- wind_solar_proj_2022_wo_rbc_2500 |>
  arrange(theta_private, gamma, xf_max_cpr) |>
  group_by(gamma) |> 
  rowwise() |> 
  mutate(R_f_private_equilibrium_cpr = 
           compute_R_value_gamma(f_private_equilibrium, gamma, q_i_mwh, x, r_0, alpha, beta)) |> 
  ungroup() |> 
  mutate(
    x_q_exp_p_total_costs_R = x * expected_p * q_i_mwh - total_cost
    - R_f_private_equilibrium_cpr
  )

wind_solar_proj_2022_wo_rbc_2500 <- wind_solar_proj_2022_wo_rbc_2500 |> 
  mutate(
    profit_cpr_private_contracts = 
      vectorized_pi_s(
        f           = f_private_equilibrium,
        q_i         = q_i_mwh,
        x           = x,
        gamma       = gamma,
        r_0         = r_0,
        alpha       = alpha,
        beta        = beta,
        total_costs = total_cost,
        T_values    = 0
      )
  )

marginal_flags <- wind_solar_proj_2022_wo_rbc_2500 %>%
  group_by(gamma) %>%
  filter(
    near(xf_private_equilibrium, xf_max_cpr)
  ) %>%
  slice(1) %>%
  mutate(is_marginal = TRUE) %>%
  ungroup()



wind_solar_proj_2022_wo_rbc_2500 <- wind_solar_proj_2022_wo_rbc_2500 %>%
  left_join(
    select(marginal_flags, gamma, theta_private, cumulative_capacity, is_marginal),
    by = c("gamma", "theta_private", "cumulative_capacity")
  ) %>%
  mutate(
    is_marginal = if_else(is.na(is_marginal), FALSE, is_marginal),
    cumulative_cutoff_private = cumulative_capacity <= private_equilibrium_quantity & xf_max_cpr <= xf_private_equilibrium
  )



ordered_vars <- c(
  "theta_private", "gamma", "f_c_cpr", "f_spot_cpr", "f_upper", "f_upper_message",
  "f_max_cpr", "f_private_equilibrium", "R_f_private_equilibrium_cpr",
  "xf_c_cpr", "xf_spot_cpr", "xf_upper_cpr", "xf_max_cpr", "xf_private_equilibrium",
  "private_equilibrium_quantity", "x_q_exp_p_total_costs", "x_q_exp_p_total_costs_R",
  "cumulative_production", "cumulative_capacity", "profit_cpr_private_contracts", "is_marginal",
  "cumulative_cutoff_private"
)

# Reorder dataframe
wind_solar_proj_2022_wo_rbc_2500 <- wind_solar_proj_2022_wo_rbc_2500 |> 
  select(
    everything(),              # Keeps all variables in current order...
    -all_of(ordered_vars),     # ...but temporarily removes the ones you want to reorder
    -profits_sp_no_cpr,        # temporarily remove chosen_xf to place it explicitly
    profits_sp_no_cpr,         # put chosen_xf in place
    all_of(ordered_vars)       # then add ordered vars in the desired order
  )




# STEP 1: Compute W^0 for gamma == 0
W_0_df_2500 <- wind_solar_proj_2022_wo_rbc_2500 |> 
  filter(gamma == 0, profits_sp_no_cpr >= 0) |> 
  group_by(theta_private) |> 
  summarise(W_0 = sum(x_q_exp_p_total_costs - r, na.rm = TRUE), .groups = "drop")

# STEP 2: Compute W(gamma) for each gamma
W_gamma_df_2500 <- wind_solar_proj_2022_wo_rbc_2500 |> 
  group_by(gamma, theta_private) |> 
  summarise(
    W_gamma = sum(
      if_else(cumulative_cutoff_private == TRUE, x_q_exp_p_total_costs_R, 0),
      na.rm = TRUE
    )
  ) |> 
  ungroup()


# STEP 4: Compute W(gamma) for gamma = 0
W_gamma_0_df_2500 <- W_gamma_df_2500 %>%
  filter(gamma == 0) %>%
  rename(W_gamma_0 = W_gamma) %>%
  select(theta_private, W_gamma_0)

eq_gamma_0_2500 <- theta_private_equilibrium %>%
  filter(gamma == 0) %>%
  select(theta_private, 
         equilibrium_quantity_gamma_0 = private_equilibrium_quantity,
         equilibrium_price_gamma_0 = private_equilibrium_price)


# STEP 5: Combine all metrics into one final table

welfare_dataset_wo_rbc_2500 <- theta_private_equilibrium |> 
  left_join(W_gamma_df_2500, by = c("gamma", "theta_private")) |> 
  left_join(W_gamma_0_df_2500, by = "theta_private") |> 
  left_join(W_0_df_2500, by = "theta_private") |> 
  left_join(eq_gamma_0_2500, by = "theta_private") |> 
  mutate(
    eq_quantity_private = private_equilibrium_quantity,
    eq_price_private = private_equilibrium_price,
    
    eq_quantity_ratio_percent = (eq_quantity_private / equilibrium_quantity_gamma_0) * 100,
    eq_price_ratio_percent = (eq_price_private / equilibrium_price_gamma_0) * 100,
    
    welfare_0_eur = if_else(gamma == 0, W_0, NA_real_),
    welfare_gamma_eur = W_gamma,
    welfare_gamma_ratio_percent = (W_gamma / W_gamma_0) * 100,
    welfare_gap_meur = (W_gamma_0 - W_gamma) / 1e6,
    welfare_gain_meur = (W_gamma - W_0) / 1e6
  ) |> 
  select(
    gamma, theta_private,
    eq_price_private, eq_price_ratio_percent,
    eq_quantity_private, eq_quantity_ratio_percent,
    welfare_0_eur, welfare_gamma_eur, welfare_gamma_ratio_percent,
    welfare_gap_meur, welfare_gain_meur
  ) |> 
  arrange(theta_private, gamma)



welfare_dataset_wo_rbc_2500

#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

## Profits ---------------------------------------------------------------

# Profits by gamma and theta

profits_by_gamma_theta_wo_rbc_2500 <- wind_solar_proj_2022_wo_rbc_2500 |> 
  group_by(gamma, theta_private) |> 
  filter(cumulative_cutoff_private == TRUE) |> 
  summarise(
    seller_profits_eur_wo_rbc = sum(profit_cpr_private_contracts, na.rm = TRUE),
    .groups = "drop"
  ) |> 
  ungroup() |> 
  arrange(theta_private, gamma, seller_profits_eur_wo_rbc)


profits_by_gamma_theta_wo_rbc_2500 <- profits_by_gamma_theta_wo_rbc_2500 |> 
  left_join(welfare_dataset_wo_rbc_2500 |> 
              select(gamma, theta_private, welfare_gamma_eur),
            by = c("gamma", "theta_private")) |> 
  mutate(buyer_profits_eur_wo_rbc = welfare_gamma_eur - seller_profits_eur_wo_rbc) |>
  rename(welfare_gamma_eur_wo_rbc = welfare_gamma_eur)


welfare_with_profits_2500_no_rbc <- welfare_dataset_wo_rbc_2500 %>%
  left_join(profits_by_gamma_theta_wo_rbc_2500, by = c("gamma", "theta_private"))

welfare_with_profits_2500_no_rbc <- welfare_with_profits_2500_no_rbc %>%
  mutate(
    buyer_profits_eur = welfare_gamma_eur - seller_profits_eur_wo_rbc,
    seller_profit_eur_wo_rbc_share_percent = round((seller_profits_eur_wo_rbc / welfare_gamma_eur) * 100, 2),
    buyer_profit_eur_wo_rbc_share_percent  = round((buyer_profits_eur_wo_rbc / welfare_gamma_eur) * 100, 2)
  ) %>%
  select(
    gamma, theta_private,
    welfare_gamma_eur_wo_rbc,
    seller_profits_eur_wo_rbc,
    buyer_profits_eur_wo_rbc,
    seller_profit_eur_wo_rbc_share_percent,
    buyer_profit_eur_wo_rbc_share_percent
  ) %>%
  arrange(theta_private, gamma)


## Welfare/Profits Comparison (RBC vs. No-RBC) ---------------------------

comparison_rbc_vs_no_rbc <- welfare_with_profits_2500_no_rbc %>%
  left_join(profits_rbc, by = c("gamma" = "gamma", "theta_private" = "theta")) |> 
  select(-theta_private) |> 
  mutate(welfare_ratio        = welfare_gamma_eur_rbc  / welfare_gamma_eur_wo_rbc,
         seller_profit_ratio  = seller_profits_eur_rbc / seller_profits_eur_wo_rbc,
         buyer_profit_ratio   = buyer_profits_eur_rbc  / buyer_profits_eur_wo_rbc)

# Long format of dataset, to help with plotting
comparison_rbc_vs_no_rbc_long <- comparison_rbc_vs_no_rbc %>%
  select(gamma, welfare_ratio, seller_profit_ratio, buyer_profit_ratio) %>%
  pivot_longer(
    cols = c(welfare_ratio, seller_profit_ratio, buyer_profit_ratio),
    names_to = "type",
    values_to = "value") |> 
  mutate(scenario = "ratio_comparison")


# Create a small table with Delta W, Reduced R and Increased Investment

welfare_differences_rbc_vs_no_rbc <- W_gamma_rbc %>%
  left_join(welfare_with_profits_2500_no_rbc, by = c("gamma" = "gamma", "theta_rbc" = "theta_private")) %>%
  left_join(profits_rbc, by = c("gamma" = "gamma", "theta_rbc" = "theta")) |> 
  select(gamma, welfare_gamma_eur_wo_rbc, welfare_gamma_eur_rbc, W_rbc) |> 
  mutate(delta_W = welfare_gamma_eur_rbc - welfare_gamma_eur_wo_rbc,
         increased_investment = delta_W - W_rbc)


### Plotting -------------------------------------------------------------

# Fixed aesthetics for baseline (see previous graphs to take the corresponding one)

baseline_color <- theme_palette_welfare[3]  # Light green or blue
baseline_shape <- 18                        # Diamond
baseline_linetype <- "dotted"

# Welfare Ratio
df <- comparison_rbc_vs_no_rbc_long |>
  filter(type == "welfare_ratio")

ggplot(df, aes(x = gamma, y = value)) +
  geom_line(color = baseline_color, linetype = baseline_linetype, linewidth = 1) +
  geom_point(shape = baseline_shape, color = baseline_color, size = 3) +
  scale_y_continuous(labels = scales::label_number(accuracy = 0.01)) +
  labs(
    x = expression(gamma),
    y = "Welfare Ratio"
  ) +
  theme_minimal(base_size = base_s) +
  theme(
    legend.position = "none",
    panel.grid.major = element_line(color = "grey90", size = 0.2),
    panel.grid.minor = element_line(color = "grey95", size = 0.1)
  )

ggsave(
  filename = file.path(with_rbc_fig_path, "04_welfare_rbc_vs_no_rbc.pdf"),
  width = 16, height = 9, dpi = 300
)

# Seller Profit Ratio
df <- comparison_rbc_vs_no_rbc_long |>
  filter(type == "seller_profit_ratio")

ggplot(df, aes(x = gamma, y = value)) +
  geom_line(color = baseline_color, linetype = baseline_linetype, linewidth = 1) +
  geom_point(shape = baseline_shape, color = baseline_color, size = 3) +
  scale_y_continuous(labels = scales::label_number(accuracy = 0.01)) +
  labs(
    x = expression(gamma),
    y = "Seller Profit Ratio"
  ) +
  theme_minimal(base_size = base_s) +
  theme(
    legend.position = "none",
    panel.grid.major = element_line(color = "grey90", size = 0.2),
    panel.grid.minor = element_line(color = "grey95", size = 0.1)
  )

ggsave(
  filename = file.path(with_rbc_fig_path, "05_seller_profit_ratio_rbc_vs_no_rbc.pdf"),
  width = 16, height = 9, dpi = 300
)

# Buyer Profit Ratio
df <- comparison_rbc_vs_no_rbc_long |>
  filter(type == "buyer_profit_ratio")

ggplot(df, aes(x = gamma, y = value)) +
  geom_line(color = baseline_color, linetype = baseline_linetype, linewidth = 1) +
  geom_point(shape = baseline_shape, color = baseline_color, size = 3) +
  scale_y_continuous(labels = scales::label_number(accuracy = 0.01)) +
  labs(
    x = expression(gamma),
    y = "Buyer Profit Ratio"
  ) +
  theme_minimal(base_size = base_s) +
  theme(
    legend.position = "none",
    panel.grid.major = element_line(color = "grey90", size = 0.2),
    panel.grid.minor = element_line(color = "grey95", size = 0.1)
  )

ggsave(
  filename = file.path(with_rbc_fig_path, "06_buyer_profit_ratio_rbc_vs_no_rbc.pdf"),
  width = 16, height = 9, dpi = 300
)

  #  ---------------------------------------------------------------------- #

# Delete some useless variables

wind_solar_proj_2022_wo_rbc_2500 <- wind_solar_proj_2022_wo_rbc_2500 |> 
  select(-is_marginal, -cumulative_cutoff_private)


# Keep useful variables, for further tables creation

profits_rbc <- profits_rbc |> 
  select(theta, gamma, profits_private, profits_rbc, seller_profits_eur_rbc,
         buyer_profits_eur_rbc, welfare_gamma_eur_rbc)



#  ---------------------------------------------------------------------- #

# Save this in a separate workbook, for clarity

datasets_wo_rbc <- list(
  `Wind_Solar_Projects` = wind_solar_proj_2022_wo_rbc_2500,
  `Welfare wo RBC (theta = 2500)` = welfare_dataset_wo_rbc_2500,
  `Profits wo RBC (theta = 2500)` = profits_by_gamma_theta_wo_rbc_2500
)

filtered_wo_rbc <- lapply(datasets_wo_rbc, filter_by_gamma)

excel_filename_wo_rbc <- paste0("02_wind_solar_projects_wo_rbc_theta_", private_demand, ".xlsx")
output_path_wo_rbc <- file.path(with_rbc_path, excel_filename_wo_rbc)

wb_wo_rbc <- createWorkbook()

for (sheet in names(filtered_wo_rbc)) {
  addWorksheet(wb_wo_rbc, sheet)
  freezePane(wb_wo_rbc, sheet, firstActiveRow = 2, firstActiveCol = 2)
  writeData(wb_wo_rbc, sheet = sheet, x = filtered_wo_rbc[[sheet]])
}

saveWorkbook(wb_wo_rbc, file = output_path_wo_rbc, overwrite = TRUE)
#  ---------------------------------------------------------------------- #

# Save this in a separate workbook, for clarity
# We save in this file the case with RBC = 2500, and the comparison
# w/o RBC = 2500



datasets_rbc <- list(
  `Wind_Solar_Projects`                    = wind_solar_proj_2022_rbc,
  `Welfare with RBC (theta = 2500)`        = welfare_dataset_rbc,
  `Welfare wo RBC (theta = 2500)`          = welfare_dataset_wo_rbc_2500,
  `Profits wo RBC (theta = 2500)`          = profits_by_gamma_theta_wo_rbc_2500,
  `Profits with RBC (theta = 2500)`        = profits_rbc,
  `Risk Reduction (theta = 2500)`          = welfare_differences_rbc_vs_no_rbc
)

filtered_rbc <- lapply(datasets_rbc, filter_by_gamma)

excel_filename_rbc <- paste0("03_wind_solar_projects_rbc_theta_", rbc_demand, ".xlsx")
output_path_rbc <- file.path(with_rbc_path, excel_filename_rbc)

wb_rbc <- createWorkbook()

for (sheet in names(filtered_rbc)) {
  addWorksheet(wb_rbc, sheet)
  freezePane(wb_rbc, sheet, firstActiveRow = 2, firstActiveCol = 2)
  writeData(wb_rbc, sheet = sheet, x = filtered_rbc[[sheet]])
}

saveWorkbook(wb_rbc, file = output_path_rbc, overwrite = TRUE)


#  ---------------------------------------------------------------------- #
#  ---------------------------------------------------------------------- #

# Section 9: Map ---------------------------------------------------------

# Find the data in data_raw path. This raw data is the same as we used at the beginning of the script
# but contains also the geographical coordinates (latitude/longitude), 
# that we need to construct our map.

file_name <- "Wind_Solar_projects_with_coordinates_Spain_2022.xlsx"
file_path <- fs::path(data_raw, file_name)


#  ---------------------------------------------------------------------- #
# Some data preparation

wind_projects_spain_2022_coordinates <- read_excel(file_path, sheet = "Wind") |> 
  mutate(type = "Wind") |> 
  filter(`Country/Area` == "Spain", `Start year` == 2022) |> 
  clean_names() |> 
  select(country_area, project_name, start_year, 
         capacity_mw, latitude, longitude, type)

solar_projects_spain_2022_coordinates <- read_excel(file_path, sheet = "Solar") |> 
  mutate(type = "Solar") |>  
  filter(`Country/Area` == "Spain", `Start year` == 2022) |> 
  clean_names() |> 
  select(country_area, project_name, start_year, 
         capacity_mw, latitude, longitude, type)

wind_solar_proj_2022_coordinates <- 
  bind_rows(wind_projects_spain_2022_coordinates, solar_projects_spain_2022_coordinates)

#  ---------------------------------------------------------------------- #
# Some projects have the same name and capacity, but different lat/lon 
# coordinates. To distinguish them, we put a number at the end of the project 
# to facilitate the subsequent join().

wind_solar_proj_2022_coordinates <- wind_solar_proj_2022_coordinates |> 
  group_by(project_name, capacity_mw) |> 
  mutate(project_name = if(n() > 1) paste0(project_name, " ", row_number()) else project_name) |> 
  ungroup() 

wind_solar_proj_2022 <- wind_solar_proj_2022 |> 
  group_by(projectname, capacity) |> 
  mutate(projectname = if(n() > 1) paste0(projectname, " ", row_number()) else projectname) |> 
  ungroup() 

#  ---------------------------------------------------------------------- #

wind_solar_proj_2022_coordinates_qi_mwh <- wind_solar_proj_2022_coordinates |> 
  left_join(
    wind_solar_proj_2022,
    by = c("project_name" = "projectname",
           "capacity_mw" = "capacity",
           "type" = "type")
  ) |> 
  select(project_name, start_year, capacity_mw, latitude, longitude, type, q_i_mwh) |> 
  mutate(
    latitude = if_else(project_name == "Puerto Del Rosario wind farm", "36", latitude),
    longitude = if_else(project_name == "Puerto Del Rosario wind farm", "-8.55", longitude)
  )

#  ---------------------------------------------------------------------- #

# Map

# Prepare your wind projects data: convert coordinates to numeric and create an sf object.
wind_solar_projects_sf <- wind_solar_proj_2022_coordinates_qi_mwh |> 
  mutate(
    latitude = as.numeric(latitude),
    longitude = as.numeric(longitude)
    ) |> 
  st_as_sf(coords = c("longitude", "latitude"), crs = 4326)

# Get Spanish provinces with cleaned names and translated province names.
esp_prov <- esp_get_prov() |>  
  clean_names() |>  
  mutate(provincia = esp_dict_translate(ine_prov_name, "es"))

# Get Canary Islands provinces and the box that defines their inset position.
can_prov <- esp_get_can_provinces() 
can_box  <- esp_get_can_box() 

# Transform wind projects to the CRS used in esp_prov.
wind_solar_projects_sf <- st_transform(wind_solar_projects_sf, st_crs(esp_prov))

# Split by type and get top 5 of each
# Get top 5 projects overall (regardless of type)
top_projects <- wind_solar_projects_sf |>
  arrange(desc(capacity_mw)) |>
  slice(1:5)

# Create the map
map_Spain_wind_solar_proj <- ggplot() +
  # Base map layers
  geom_sf(data = esp_prov, fill = "grey99", color = "black") +
  geom_sf(data = can_prov, fill = "grey99", color = "black") +
  geom_sf(data = can_box, fill = NA, color = "black", size = 1) +
  
  # Wind projects
  geom_sf(data = filter(wind_solar_projects_sf, type == "Wind"),
          aes(size = capacity_mw, color = type),
          alpha = 0.3, show.legend = TRUE) +
  
  # Solar projects
  geom_sf(data = filter(wind_solar_projects_sf, type == "Solar"),
          aes(size = capacity_mw, color = type),
          alpha = 0.3, show.legend = TRUE) +
  
  # Labels for top 5 largest projects (optional)
  geom_label_repel(
    data = top_projects,
    aes(label = project_name, geometry = geometry, color = type),
    stat = "sf_coordinates",
    fill = "white", alpha = 0.9,
    size = 3, label.size = 0,
    box.padding = 1.5, point.padding = 0.5,
    min.segment.length = 0,
    segment.size = 0.3,
    show.legend = FALSE
  ) +
  
  # Size scale (auto-scaled using sqrt transform)
  scale_size_continuous(
    trans = "sqrt",                # Helps small projects show up
    range = c(2, 18),              # Controls dot size visually
    name = "Capacity (MW)"         # Legend title
  ) +
  
  # Project type color
  scale_color_manual(
    values = theme_palette_map, # #084594 (dark blue paper), #9ecae1 (light blue paper), #002d18 (dark green), #6ecf87 (light green)
    name = "Project Type"
  ) +
  
  theme_void() +
  theme(
    legend.position = "right",
    legend.title = element_text(size = 10, face = "bold"),
    legend.text = element_text(size = 9)
  )

# Save the plot
plot_filename <- paste0("spain_map_wind_solar_proj_2022", file_suffix, ".pdf")
plot_path_cpr <- file.path(out_figures, plot_filename)

ggsave(
  filename = plot_path_cpr,
  plot = map_Spain_wind_solar_proj,
  width = 16,
  height = 9,
  dpi = 300
)


# ------------------------------------------------------------------------ #
# END OF THE SCRIPT
# ------------------------------------------------------------------------ #

