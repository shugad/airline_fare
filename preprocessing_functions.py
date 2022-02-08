import re


# function for month extraction
def get_season(month):
    if month in [12, 1, 2]:
        return "winter"
    elif month in [3, 4, 5]:
        return "spring"
    elif month in [6, 7, 8]:
        return "summer"
    else:
        return "fall"


# function for period extraction
def get_period_of_day(x):
    x = int(x[:2])
    if (x > 4) and (x <= 8):
        return 'Early Morning'
    elif (x > 8) and (x <= 12):
        return 'Morning'
    elif (x > 12) and (x <= 16):
        return'Noon'
    elif (x > 16) and (x <= 20):
        return 'Eve'
    elif (x > 20) and (x <= 24):
        return'Night'
    elif x <= 4:
        return'Late Night'


# function for duration counting
def get_duration(x):
    replacements = [
     ('h', ':'),
     ('m', ''),
     (' ', '')]
    for old, new in replacements:
        x = re.sub(old, new, x)
    splt = x.split(':')
    hours_to_min = int(splt[0])*60
    if len(splt) == 2 and splt[1].isdigit():
        fin = hours_to_min + int(splt[1])
    else:
        fin = hours_to_min
    return fin


# function for duration counting
def get_duration_hours(x):
    replacements = [
     ('h', ':'),
     ('m', ''),
     (' ', '')]
    for old, new in replacements:
        x = re.sub(old, new, x)
    return x.split(':')[0]


# map value with respective dict values
def get_mapped_value(value):
    return dict_groups_airlines[value]


dict_groups_airlines = {
    'multiple carriers premium economy': 'premium', 'jet airways business': 'premium',
    'jet airways': 'premium', 'vistara': 'medium', 'vistara premium economy': 'medium',
    'goair': 'medium', 'multiple carriers': 'medium', 'air india': 'medium',
    'trujet': 'low-cost', 'spicejet': 'low-cost', 'indigo': 'low-cost', 'air asia': 'low-cost'
}


