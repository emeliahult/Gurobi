"""
This file is the mathematical model of the Nurse scheduling problem
Created by Anders N. Gullhav 2020-09-29 for Mosel Xpress and adapted for Gurobi by Brede Sørøy 2024-09-12
Anders N. Gullhav, IØT, anders.gullhav@ntnu.no
"""
import math

import gurobipy as gp
from gurobipy import GRB
import pandas as pd

from tabulate import tabulate


instance_file = "NurseSchedulingData.xlsx"

verbose = True      # True: print, False: no print

original = True     # True: use standard settings, False: do not use standard settings

presolve = -1       # -1: automatic, 0: off, 1: conservative, 2: aggressive
heuristics = 0.05   # the value as the desired fraction of total MIP runtime devoted to heuristics
branching = -1      # -1: automatic, 0: pseudo reduced cost, 1: pseudo shadow price, 2: maximum infeasibility, 3: strong
cuts = -1           # -1: automatic, 0: off, 1: moderate, 2: aggressive, 3: very aggressive


def get_demand(file_path):
    return gp.tupledict(pd.read_excel(file_path, index_col=0, sheet_name="Demand").stack())


def get_external_costs(file_path):
    return gp.tupledict(pd.read_excel(file_path, index_col=0, sheet_name="CostExt").stack())


def get_wishes(file_path):
    return gp.tupledict(pd.read_excel(file_path, index_col=0, sheet_name="Wishes").stack())


def get_amount_values(file_path):
    return pd.read_excel(file_path, sheet_name="N_values").values.tolist()[0]


def get_shift_indices(file_path):
    return tuple(pd.read_excel(file_path, sheet_name="Shifts").columns.tolist())


def get_min_off_days_and_max_night_shifts(file_path):
    temp_df = pd.read_excel(file_path, index_col=0, sheet_name="MinOffDaysAndMaxNightShifts")
    return gp.multidict(zip(temp_df.index, temp_df.values))


model = gp.Model("NurseScheduling")

n_nurses, n_days, max_days_in_row = get_amount_values(instance_file)

nurses = tuple(range(1, n_nurses + 1))                  # N
days = tuple(range(1, n_days + 1))                      # D
saturdays = tuple(day for day in days if day % 7 == 6)  # D^Sat
shifts = get_shift_indices(instance_file)               # S

cost_ext = get_external_costs(instance_file)            # C
demand = get_demand(instance_file)                      # A

_, min_off_days, max_night_shifts = get_min_off_days_and_max_night_shifts(instance_file)    # F and S^MaxNights
wishes = get_wishes(instance_file)                      # W

# compute upper bound for external workers on each shift and day
ub_for_external = gp.tupledict({(s, d): demand[s, d]//2 for s in shifts for d in days})

# compute big_m_wishes and big_m_off_days
big_m_wishes = gp.tupledict({n: math.ceil(0.5*sum(wishes[n, d] for d in days)) for n in nurses})    # M_W
big_m_off_days = gp.tupledict({n: min_off_days[n] + 2 for n in nurses})                             # M_F

assignment = model.addVars(nurses, shifts, days,
                           vtype=GRB.BINARY, name="nurse_assigned_to_shift_on_day")     # x
off_day = model.addVars(nurses, days,
                        vtype=GRB.BINARY, name="nurse_has_off_day")                     # z
external = model.addVars(shifts, days, ub=ub_for_external,
                         vtype=GRB.INTEGER, name="externals_assigned_to_shift_on_day")  # y

# TODO create extra_off_day ("delta") variables and make them binary

extra_day_off = model.addVars(nurses, 
                      vtype = GRB.BINARY, name = "delta")

total_cost = external.prod(cost_ext)

model.setObjective(total_cost, sense=GRB.MINIMIZE)  # minimisation is default but this enhances readability

demand_cover = model.addConstrs(
    (
        assignment.sum("*", shift, day) + external[shift, day] == demand[shift, day]
        for shift in shifts for day in days
    ),
    name="CoverDemand")

one_shift_day = model.addConstrs(
    (
        assignment.sum(nurse, "*", day) + off_day[nurse, day] == 1
        for nurse in nurses for day in days
    ),
    name="OneShiftPerDay")


# TODO implement additional constraints here
#  Hint: Specific shift indices (strings) can be set by encapsulating the string with ' '.
#        The assignment variables for evening shift 'E' can be written like this: assignment[nurse,'E',day]


rest_ed = model.addConstrs( # 1a) A nurse should have at least 11 hours of rest between shifts 
(
    assignment[nurse, 'E', day] + assignment[nurse, 'D', day + 1] <= 1
    for nurse in nurses for day in days if day < n_days
), 
name = "RestEd")


consecutive_weekend = model.addConstrs( #3) A nurse should not work two weekends in a row. 
    
    (
        off_day[nurse, day] + off_day[nurse, day + 7] >= 1
        for nurse in nurses for day in saturdays if day < n_days - 7
    ), 
    name = "ConsecutiveWeekend")

days_in_row = model.addConstrs( #4) A nurse should not work more than M days in a row. 
    (
        gp.quicksum(off_day[nurse, temp_day] for temp_day in range(day, day + max_days_in_row + 1)) >= 1 
        for nurse in nurses for day in days if day <= n_days - max_days_in_row
    ), 
    name = "DaysInRow")

tot_off_days = model.addConstrs( #5) Each nurse should have at least F days off during the planning period
    (
        off_day.sum(nurse, "*") >= min_off_days[nurse]
        for nurse in nurses
    ), 
    name = "EnsureTotalOffDays")
    



model.Params.LogToConsole = verbose

if not original:

    model.Params.Presolve = presolve
    model.Params.Heuristics = heuristics
    model.Params.VarBranch = branching
    model.Params.Cuts = cuts


model.optimize()


def find_assigned_shift_for_nurse_on_day(nurse, day, *, off_day_indicator=''):
    if assignment[nurse, 'D', day].X > 1e-5:
        return 'D'
    elif assignment[nurse, 'E', day].X > 1e-5:
        return 'E'
    elif assignment[nurse, 'N', day].X > 1e-5:
        return 'N'
    else:
        return off_day_indicator


if model.Status == GRB.INFEASIBLE:
    print("\nThe model is infeasible")
else:
    print(f"\nOptimal objective value: {model.ObjVal:.2f}\n")
    print("NURSE SCHEDULE\n")

    table = list([nurse] + list(find_assigned_shift_for_nurse_on_day(nurse, day, off_day_indicator="F") for day in days) for nurse in nurses)

    for shift in shifts:
        table.append([f"Ext {shift}"] + list(f"{external[shift, day].X:.0f}" if external[shift, day].X > 1e-5 else "" for day in days))

    print(" " * 9 + "Day")

    print(tabulate(table, headers=["Nurse", *days], stralign="center", numalign="center", tablefmt="rst"))

    print("\nSUM OF OFF DAYS:")

    for nurse in nurses:
        print(f"Nurse {nurse:02d}: {sum(off_day[nurse, day].X for day in days):.0f}")

    print("\nFULFILLED WISHES:")

    for nurse in nurses:
        print(f"Nurse {nurse:02d}: {sum(wishes[nurse, day] * off_day[nurse, day].X for day in days):.0f} fulfilled wishes")
