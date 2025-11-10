
from schedule import Schedule
import random, math, copy
from solver_naive import solve as naive_solution


def trouver_sol_voisine(courses, solution, conflicts) : 
    num_neighbors = 5
    candidates = []
    for _ in range(num_neighbors):
        c = random.choice(courses)
        new_slot = random.choice(list(set(solution.values())) + [max(solution.values()) + 1])
        new_solution = copy.deepcopy(solution)
        new_solution[c] = new_slot
        n_conflicts = sum(new_solution[a] == new_solution[b] for a, b in conflicts)
        cost = len(set(new_solution.values())) + 10 * n_conflicts
        candidates.append((c, new_solution, cost))
        
    course, new_solution, cost = min(candidates, key=lambda x: x[2])
    return(course, new_solution,cost)

def solve(schedule):
    """
    Your solution of the problem
    :param schedule: object describing the input
    :return: a list of tuples of the form (c,t) where c is a course and t a time slot. 
    """
    # Add here your agent

    #Premierement, on trouve une premiere solution Gloutonne
    courses= list(schedule.course_list)
    conflicts = list(schedule.conflict_list)
    #cours_tri = sorted(courses, key=lambda x:len(schedule.get_node_conflicts(x)), reverse=True)
    # for course in cours_tri:
    #     assigned = False
    #     for t in range(1, len(courses) + 1):
    #         if all(solution.get(nei, 0) != t for nei in schedule.get_node_conflicts(course)):
    #             solution[course] = t
    #             assigned = True
    #             break
    #     if not assigned:
    #         solution[course] = len(set(solution.values())) + 1
    # best_solution = copy.deepcopy(solution)
    # best_cost = schedule.get_n_creneaux(best_solution)

    solution = naive_solution(schedule)
    best_solution = copy.deepcopy(solution)
    best_cost = schedule.get_n_creneaux(best_solution)

    # Recherche de meilleure solution.
     
    T = 1.0          # température initiale
    Tmin = 1e-3       # température minimale
    alpha = 0.995     # facteur de refroidissement
    max_iter = 10000

    for _ in range(max_iter):
        # Choisir un cours au hasard


        course,_,_ = trouver_sol_voisine(courses, solution, conflicts)

        # Choisir un nouveau créneau possible (ou en créer un nouveau)
        possible_slots = list(set(solution.values()))
        new_slot = random.choice(possible_slots + [max(possible_slots) + 1])

        # Créer une solution voisine
        new_solution = copy.deepcopy(solution)
        new_solution[course] = new_slot

        # Évaluer la solution voisine
        n_conflicts = sum(new_solution[a] == new_solution[b] for a, b in conflicts)
        cost = len(set(new_solution.values())) + 10 * n_conflicts 

        n_conflicts_best = sum(best_solution[a] == best_solution[b] for a, b in conflicts)
        best_cost = len(set(best_solution.values())) + 10 * n_conflicts_best

        # Savoir si on garde cette solution ou non
        if cost < best_cost or random.random() < math.exp((best_cost - cost) / T):
            solution = new_solution
            if n_conflicts == 0:
                best_solution = copy.deepcopy(new_solution)

        T*= alpha
        if T < Tmin:
            break

    return best_solution 
