"""
CSP-based Sudoku Solver
AI2002 - Artificial Intelligence, Assignment 5
FAST-NUCES Chiniot-Faisalabad

Techniques used:
  - AC-3 (Arc Consistency Algorithm 3) for initial domain reduction
  - Backtracking Search with Forward Checking
  - MRV (Minimum Remaining Values) heuristic for variable selection
  - LCV (Least Constraining Value) heuristic for value ordering
"""

import sys
import copy

# ─────────────────────────────────────────────
# Board Reading
# ─────────────────────────────────────────────

def read_board(filename):
    """Read a 9x9 Sudoku board from a text file. 0 = empty cell."""
    board = []
    with open(filename) as f:
        for line in f:
            row = [int(c) for c in line.strip() if c.isdigit()]
            if row:
                board.append(row)
    assert len(board) == 9 and all(len(r) == 9 for r in board), \
        "Board must be 9x9"
    return board

# ─────────────────────────────────────────────
# Peer (neighbor) computation
# ─────────────────────────────────────────────

def get_peers(r, c):
    """
    Return the set of all (row, col) positions that share a row,
    column, or 3x3 box with (r, c) — excluding (r, c) itself.
    """
    peers = set()

    # Same row
    for col in range(9):
        if col != c:
            peers.add((r, col))

    # Same column
    for row in range(9):
        if row != r:
            peers.add((row, c))

    # Same 3x3 box
    box_r, box_c = 3 * (r // 3), 3 * (c // 3)
    for row in range(box_r, box_r + 3):
        for col in range(box_c, box_c + 3):
            if (row, col) != (r, c):
                peers.add((row, col))

    return peers


# Precompute all peers once (used throughout)
PEERS = {(r, c): get_peers(r, c) for r in range(9) for c in range(9)}


# ─────────────────────────────────────────────
# Domain Initialisation
# ─────────────────────────────────────────────

def init_domains(board):
    """
    Build the initial domain dict.
    - Assigned cells get domain {digit}.
    - Empty cells start with {1..9} minus values already seen in peers.
    """
    domains = {}
    for r in range(9):
        for c in range(9):
            if board[r][c] != 0:
                domains[(r, c)] = {board[r][c]}
            else:
                used = {board[pr][pc]
                        for (pr, pc) in PEERS[(r, c)]
                        if board[pr][pc] != 0}
                domains[(r, c)] = set(range(1, 10)) - used
    return domains


# ─────────────────────────────────────────────
# AC-3 (Arc Consistency)
# ─────────────────────────────────────────────

def ac3(domains):
    """
    Run AC-3 to enforce arc consistency.
    Returns False if a domain is wiped out (puzzle unsolvable),
    True otherwise.
    Modifies domains in-place.
    """
    # Queue of all arcs (Xi, Xj) where Xi and Xj are peers
    queue = []
    for cell in domains:
        for peer in PEERS[cell]:
            queue.append((cell, peer))

    while queue:
        xi, xj = queue.pop(0)

        if revise(domains, xi, xj):
            if len(domains[xi]) == 0:
                return False   # Domain wiped out — contradiction

            # Re-check all arcs pointing TO xi (since xi's domain shrank)
            for xk in PEERS[xi]:
                if xk != xj:
                    queue.append((xk, xi))

    return True


def revise(domains, xi, xj):
    """
    Remove values from domains[xi] that have no valid support in domains[xj].
    Because Sudoku constraints are 'all-different', a value v in xi
    has support in xj only if xj's domain contains at least one value ≠ v.
    Returns True if any value was removed.
    """
    revised = False
    for v in list(domains[xi]):
        # v has no support if xj's entire domain is {v}
        if domains[xj] == {v}:
            domains[xi].discard(v)
            revised = True
    return revised


# ─────────────────────────────────────────────
# Heuristics
# ─────────────────────────────────────────────

def select_unassigned_variable(domains):
    """
    MRV heuristic: choose the unassigned cell with the fewest
    remaining legal values (minimum domain size > 1 means unassigned
    — or rather, cells whose domain still has > 0 and the board value is 0).
    We treat cells with domain size == 1 as 'assigned'.
    
    Note: We only branch on cells that are genuinely unassigned.
    """
    unassigned = [(len(d), cell)
                  for cell, d in domains.items()
                  if len(d) > 1]
    if not unassigned:
        return None
    _, cell = min(unassigned)
    return cell


def order_domain_values(cell, domains):
    """
    LCV heuristic: order values by how few choices they eliminate
    for neighboring cells (ascending constraint count).
    """
    def count_constraints(val):
        count = 0
        for peer in PEERS[cell]:
            if val in domains[peer] and len(domains[peer]) > 1:
                count += 1
        return count

    return sorted(domains[cell], key=count_constraints)


# ─────────────────────────────────────────────
# Forward Checking (used inside backtrack)
# ─────────────────────────────────────────────

def forward_check(cell, value, domains):
    """
    After assigning `value` to `cell`, remove `value` from all peers' domains.
    Returns (True, domains) if consistent, (False, None) if any peer's domain
    is wiped out.
    """
    new_domains = copy.deepcopy(domains)
    new_domains[cell] = {value}

    for peer in PEERS[cell]:
        if len(new_domains[peer]) > 1:           # peer not yet assigned
            new_domains[peer].discard(value)
            if len(new_domains[peer]) == 0:
                return False, None               # wipeout

    return True, new_domains


# ─────────────────────────────────────────────
# Backtracking Search
# ─────────────────────────────────────────────

# Global counters
backtrack_calls = 0
backtrack_failures = 0


def is_complete(domains):
    """A solution is complete when every cell's domain has exactly one value."""
    return all(len(d) == 1 for d in domains.values())


def backtrack(domains):
    """
    Recursive backtracking search with forward checking.
    Returns a solved domains dict or None on failure.
    """
    global backtrack_calls, backtrack_failures
    backtrack_calls += 1

    if is_complete(domains):
        return domains  # Solution found!

    cell = select_unassigned_variable(domains)
    if cell is None:
        return None

    for value in order_domain_values(cell, domains):
        ok, new_domains = forward_check(cell, value, domains)

        if ok:
            # Optionally run AC-3 for additional propagation
            if ac3(new_domains):
                result = backtrack(new_domains)
                if result is not None:
                    return result

    # No value worked — backtrack
    backtrack_failures += 1
    return None


# ─────────────────────────────────────────────
# Solving Entry Point
# ─────────────────────────────────────────────

def solve(board):
    """
    Full solve pipeline:
    1. Initialise domains
    2. Run AC-3 for initial propagation
    3. Run backtracking search
    Returns (solution_board, bt_calls, bt_failures) or (None, ...)
    """
    global backtrack_calls, backtrack_failures
    backtrack_calls = 0
    backtrack_failures = 0

    domains = init_domains(board)

    # Initial AC-3 pass
    if not ac3(domains):
        return None, backtrack_calls, backtrack_failures

    # Backtracking
    result = backtrack(domains)

    if result is None:
        return None, backtrack_calls, backtrack_failures

    # Convert domains → 2D board
    solution = [[list(result[(r, c)])[0] for c in range(9)] for r in range(9)]
    return solution, backtrack_calls, backtrack_failures


# ─────────────────────────────────────────────
# Display Helpers
# ─────────────────────────────────────────────

def print_board(board, title=""):
    if title:
        print(f"\n{'─'*37}")
        print(f"  {title}")
        print(f"{'─'*37}")
    for i, row in enumerate(board):
        if i % 3 == 0 and i != 0:
            print("  ------+-------+------")
        line = ""
        for j, val in enumerate(row):
            if j % 3 == 0 and j != 0:
                line += " | "
            line += (" " if j % 3 != 0 else " ") + str(val)
        print(line)


def validate(board):
    """Verify every row, col and box contains digits 1-9 exactly once."""
    digits = set(range(1, 10))
    # Rows
    for r in range(9):
        if set(board[r]) != digits:
            return False
    # Cols
    for c in range(9):
        if {board[r][c] for r in range(9)} != digits:
            return False
    # Boxes
    for br in range(3):
        for bc in range(3):
            box = {board[br*3+r][bc*3+c]
                   for r in range(3) for c in range(3)}
            if box != digits:
                return False
    return True


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    puzzles = [
        ("easy.txt",     "Easy"),
        ("medium.txt",   "Medium"),
        ("hard.txt",     "Hard"),
        ("veryhard.txt", "Very Hard"),
    ]

    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("\n" + "="*50)
    print("   CSP Sudoku Solver — AI2002 Assignment 5")
    print("="*50)

    for filename, label in puzzles:
        path = os.path.join(script_dir, filename)
        board = read_board(path)

        print_board(board, f"{label} — Input ({filename})")

        solution, calls, failures = solve(board)

        if solution is None:
            print(f"\n  *** No solution found for {label}! ***")
        else:
            valid = validate(solution)
            print_board(solution, f"{label} — Solution  [valid: {valid}]")
            print(f"\n  Backtrack calls   : {calls}")
            print(f"  Backtrack failures: {failures}")

            # Brief comment on the numbers
            if calls <= 10:
                comment = "Very few backtracks — AC-3 + MRV solved it almost directly."
            elif calls <= 100:
                comment = "Moderate search — forward checking pruned most bad paths."
            elif calls <= 1000:
                comment = "Harder puzzle — more exploration needed despite good heuristics."
            else:
                comment = "Very hard puzzle — deep backtracking required; heuristics help but cannot avoid all dead-ends."
            print(f"  Comment: {comment}")

        print()


if __name__ == "__main__":
    main()
