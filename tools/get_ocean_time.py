from datetime import datetime, timedelta

def seconds_to_datetime(seconds):
    # Define the base date and time (1.1.2000 00:00)
    base_date = datetime(2000, 1, 1, 0, 0, 0)

    # Calculate the target date by adding the elapsed seconds to the base date
    target_date = base_date + timedelta(seconds=seconds)

    return target_date

def datetime_to_seconds(target_date):
    # Define the base date and time (1.1.2000 00:00)
    base_date = datetime(2000, 1, 1, 0, 0, 0)

    # Calculate the difference in seconds
    seconds_elapsed = (target_date - base_date).total_seconds()

    return seconds_elapsed

# Example usage:
target_date = datetime(2016, 3, 3, 0, 0, 0)  # Replace with your desired date and time
result = datetime_to_seconds(target_date)

print("Target Date and Time:", target_date)
print("Elapsed Seconds:", result)
print("Elapsed hours:", result/3600)


# Example usage:
seconds_elapsed = 123456789  # Replace with your actual number of seconds
result = seconds_to_datetime(seconds_elapsed)

print("Elapsed Seconds:", seconds_elapsed)
print("Target Date and Time:", result)

