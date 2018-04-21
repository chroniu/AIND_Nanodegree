from utils import *

row_units = [cross(r, cols) for r in rows]
column_units = [cross(rows, c) for c in cols]
square_units = [cross(rs, cs) for rs in ('ABC','DEF','GHI') for cs in ('123','456','789')]
unitlist = row_units + column_units + square_units


# TODO: Update the unit list to add the new diagonal units
diagonal_units = [[x+y for x,y in zip(rows, cols)]] + [[ x+y for x,y in zip(rows, cols[::-1])]]
naked_list = row_units + column_units #+ diagonal_units

unitlist = unitlist + diagonal_units

units = dict((s, [u for u in unitlist if s in u]) for s in boxes)
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in boxes)


import pdb

def square_number(unit_key):
    number = None
    if(unit_key[0] in "ABC"):
        number = 0
    elif(unit_key[0] in "DEF"):
        number = 3
    else:
        number = 6
    return number + (int(unit_key[1])-1) // 3


def naked_twins(values):
    """Eliminate values using the naked twins strategy.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict
        The values dictionary with the naked twins eliminated from peers

    Notes
    -----
    Your solution can either process all pairs of naked twins from the input once,
    or it can continue processing pairs of naked twins until there are no such
    pairs remaining -- the project assistant test suite will accept either
    convention. However, it will not accept code that does not process all pairs
    of naked twins from the original input. (For example, if you start processing
    pairs of twins and eliminate another pair of twins before the second pair
    is processed then your code will fail the PA test suite.)

    The first convention is preferred for consistency with the other strategies,
    and because it is simpler (since the reduce_puzzle function already calls this
    strategy repeatedly).
    """

    # TODO: Implement this function!
    pairs = []
    for unit in naked_list:
        sub_list = [x for x in unit if len(values[x]) == 2]

        for i in range(len(sub_list) -1):
            for j in range(i+1, len(sub_list)):
                if values[sub_list[i]] == values[sub_list[j]]:
                    pairs += [(sub_list[i], sub_list[j])]

    for x, y in pairs:
        units_clean = []
        if x[0] == y[0]: # same row
            units_clean = row_units[ord(x[0]) - 65].copy()
        else: # same column
            units_clean += column_units[int(x[1]) -1].copy()

        if(square_number(x) == square_number(y)):
            units_clean += square_units[square_number(x)].copy()

        for element in units_clean:

            if(element in [x, y]):
                continue
            for digit in values[x]:
                if(digit in values[element]):
                    values[element] = values[element].replace(digit, "")

    return values


def eliminate(values):
    """Apply the eliminate strategy to a Sudoku puzzle

    The eliminate strategy says that if a box has a value assigned, then none
    of the peers of that box can have the same value.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict
        The values dictionary with the assigned values eliminated from peers
    """
    for unit in values.keys():
        if len(values[unit])==1:
            values[unit] = values[unit]
            for key in peers[unit]:
                ## # print (unit, key)
                values[key] = values[key].replace(values[unit],"")
        else:
            values[unit] = values[unit]
        # # # print(result[unit])
    return values


def only_choice(values):
    """Apply the only choice strategy to a Sudoku puzzle

    The only choice strategy says that if only one box in a unit allows a certain
    digit, then that box must be assigned that digit.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict
        The values dictionary with all single-valued boxes assigned

    Notes
    -----
    You should be able to complete this function by copying your code from the classroom
    """
    # TODO: Copy your code from the classroom to complete this function
    digits = "123456789"

    for unit in unitlist: # ok, the code of the sample is a lot better than the one that I tought
        for index in digits:
            dplaces = [box for box in unit if index in values[box]]
            if len(dplaces) == 1:
                values[dplaces[0]] = index
    return values





def reduce_puzzle(values):
    """Reduce a Sudoku puzzle by repeatedly applying all constraint strategies

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict or False
        The values dictionary after continued application of the constraint strategies
        no longer produces any changes, or False if the puzzle is unsolvable
    """
    stalled = False
    while not stalled:
        # Check how many boxes have a determined value
        solved_values_before = len([box for box in values.keys() if len(values[box]) == 1])

        # Your code here: Use the Eliminate Strategy
        eliminate(values)
        # Your code here: Use the Only Choice Strategy
        only_choice(values)
        # Check how many boxes have a determined value, to compare
        solved_values_after = len([box for box in values.keys() if len(values[box]) == 1])
        # If no new values were added, stop the loop.
        stalled = solved_values_before == solved_values_after
        # Sanity check, return False if there is a box with zero available values:
        if len([box for box in values.keys() if len(values[box]) == 0]):
            return False
    return values


def search(values):
    """Apply depth first search to solve Sudoku puzzles in order to solve puzzles
    that cannot be solved by repeated reduction alone.

    Parameters
    ----------
    values(dict)
        a dictionary of the form {'box_name': '123456789', ...}

    Returns
    -------
    dict or False
        The values dictionary with all boxes assigned or False

    Notes
    -----
    You should be able to complete this function by copying your code from the classroom
    and extending it to call the naked twins strategy.
    """
    "Using depth-first search and propagation, create a search tree and solve the sudoku."
    # First, reduce the puzzle using the previous function
    values = reduce_puzzle(values)

    if not values:
        return False
    if all(len(values[s]) == 1 for s in boxes):
        return values

    values = naked_twins(values)

    # Choose one of the unfilled squares with the fewest possibilities
    # for key, value in values.items():
    _, key = min((len(values[s]), s) for s in boxes if len(values[s]) > 1)

    for value in values[key]:
        values_copy = values.copy()
        values_copy[key] = value

        values_copy = search(values_copy)
        if(values_copy):
            return values_copy

    return False


def solve(grid):
    """Find the solution to a Sudoku puzzle using search and constraint propagation

    Parameters
    ----------
    grid(string)
        a string representing a sudoku grid.

        Ex. '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'

    Returns
    -------
    dict or False
        The dictionary representation of the final sudoku grid or False if no solution exists.
    """
    values = grid2values(grid)
    values = search(values)
    return values


if __name__ == "__main__":
    diag_sudoku_grid = '2.............62....1....7...6..8...3...9...7...6..4...4....8....52.............3'
    display(grid2values(diag_sudoku_grid))
    result = solve(diag_sudoku_grid)
    display(result)

    try:
        import PySudoku
        PySudoku.play(grid2values(diag_sudoku_grid), result, history)

    except SystemExit:
        pass
    except:
        print('We could not visualize your board due to a pygame issue. Not a problem! It is not a requirement.')
