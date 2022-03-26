import pulp

# Eingabe und Konstanten:
# =======================
# Dies ist die Flowshop-Eingabe: 4 Jobs auf 3 Maschinen mit fester Technologiereihenfolge (Maschine1->Maschine2->Maschine3)
jobs = [[[1, 5], [2, 7], [3, 5], 21], [[1, 4], [2, 6], [3, 7], 18], [[1, 3], [2, 6], [3, 6], 21], [[1, 5], [2, 5], [3, 4], 19]]
n_machines = 3
# processing_time[i,k]: Bearbeitungszeit der Operation des Jobs i auf der Maschine k
processing_time = {(i + 1, jobs[i][k][0]): jobs[i][k][1] for i in range(len(jobs)) for k in range(n_machines)}
# due_date[i]: Soll-Fertigstellungstermin (Due Date) des Jobs i
due_date = {i + 1: jobs[i][3] for i in range(len(jobs))}
M = 20000  # für die Big-M-Methode


# Entscheidungsvariablen
# =======================
# start_time[i,k]: Startzeitpunkt einer Operation (Job i auf auf Maschine k)
start_time = pulp.LpVariable.dicts(
    "start_time", indexs=(range(1, len(jobs) + 1), range(1, n_machines + 1)), cat=pulp.LpInteger, lowBound=0
)
# predecessor[i,j,k]: 1, wenn Job i ein Vorgänger von Job j auf Maschine k ist. Sonst 0
predecessor = pulp.LpVariable.dicts(
    "predecessor", indexs=(range(1, len(jobs) + 1), range(len(jobs) + 1), range(1, n_machines + 1)), cat=pulp.LpBinary
)
# tardiness[i]: Verspätung eines Jobs = Anzahl Zeiteinheiten, die das Due Date eines Jobs i überschritten wurde
# Wichtig: nur positive Werte sind hier zulässig! Verfrühte Fertigstellungen sollen nicht berücksichtigt werden.
tardiness = pulp.LpVariable.dicts("tardiness", indexs=(range(1, len(jobs) + 1)), lowBound=0)


# Zielfunktion
# =======================
# Das Optimierungsziel ist die Minimierung der Gesamtverspätung (min sum tardiness).
# Bedeutet: Die Jobs müssen so angeordnet werden, sodass überschrittene Due Dates minimiert werden sollen.
scheduling_objective = pulp.LpVariable("Best solution (min sum. tardiness)", lowBound=0, cat=pulp.LpInteger)
model = pulp.LpProblem("FSSP min tardiness", pulp.LpMinimize)
model += scheduling_objective
# Die Summe aller Verspätungen ist folglich zu minimieren
model += pulp.lpSum([tardiness[j] for j in range(1, len(jobs) + 1)]) <= scheduling_objective


# Nebenbedingungen
# =======================
# Operationen eines Jobs dürfen frühestens dann starten, wenn die unmittelbare Vorgängeroperation abgeschlossen ist.
# Dieser Constraint ist bereits korrekt definiert und muss nicht mehr angepasst werden.
for i in range(1, len(jobs) + 1):
    for l in range(1, n_machines):
        model += start_time[i][l] + processing_time[i, l] <= start_time[i][l + 1]

# Eine Maschine darf nur eine Operation gleichzeitig bearbeiten (Einsatz der Big-M-Methode)
# Dieser Constraint ist bereits korrekt definiert und muss nicht mehr angepasst werden.
for j in range(1, len(jobs) + 1):
    for i in range(1, len(jobs) + 1):
        if i == j:
            continue
        for k in range(1, n_machines + 1):
            model += (M + processing_time[j, k]) * predecessor[i][j][k] + (start_time[i][k] - start_time[j][k]) >= processing_time[j, k]
            model += (M + processing_time[i, k]) * (1 - predecessor[i][j][k]) + (start_time[j][k] - start_time[i][k]) >= processing_time[i, k]

# TODO: Hier die fehlenden Constraints zur Vervollständigung des Optimierungsmodells ergänzen
# Info: Wenn der Solver ein globales Optimum von 20 errechnet, so ist dies ein Indiz für korrekt definierte Nebenbedingungen.

for j in range(1, len(jobs) + 1):
    model += tardiness[j] >= start_time[j][n_machines] + processing_time[j, n_machines] - due_date[j]
    model += tardiness[j] >= 0


# Berechnung und Ausgabe
# =======================
model.solve(pulp.PULP_CBC_CMD(msg=1))
print("VARIABLES SET\n=============")
for var in model.variables():
    print(f"{var.name}: {var.varValue}")
