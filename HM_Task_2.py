# Task 2  I created this model basing on random parts of dataset, accuracy of it is terrible,
# but it was not part of the task to make it good
# It is in diffrent file so we can use it without model creation file
def heuristic_model(Elevation, Aspect):
    if Elevation > 2850:
        return 1
    elif Elevation in range(2700, 2850):
        return 2
    elif Elevation in range(1800, 2550) and Aspect > 115:
        return 3
    elif Elevation in range(2000, 2200) and Aspect < 100:
        return 4
    elif Elevation in range(2200, 2700) and Aspect < 100:
        return 5
    elif Elevation in range(2200, 2400) and Aspect > 200:
        return 6
    else:
        return 7