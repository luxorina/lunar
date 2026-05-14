"""
Sun and Moon Position Plotter
Plots the position of the sun and moon in the sky with configurable date/time and location
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
from timezonefinder import TimezoneFinder

# You'll need to install astropy and timezonefinder:
# pip install astropy timezonefinder pytz

from astropy.coordinates import EarthLocation, AltAz, get_body
from astropy.time import Time
import astropy.units as u


def plot_sun_moon_position(start_date, start_time, end_date, end_time, latitude=9.338212, longitude=-82.258937):
    """
    Plot the sun and moon's position in the sky for a given time range and location.
    
    Parameters:
    -----------
    start_date : str
        Start date in format 'YYYY-MM-DD' (e.g., '2025-04-10')
    start_time : str
        Start time in format 'HH:MM:SS' (e.g., '06:00:00') - interpreted as LOCAL time
    end_date : str
        End date in format 'YYYY-MM-DD' (e.g., '2025-04-11')
    end_time : str
        End time in format 'HH:MM:SS' (e.g., '18:00:00') - interpreted as LOCAL time
    latitude : float
        Observer latitude in decimal degrees (default: Costa Rica at 9.338212)
    longitude : float
        Observer longitude in decimal degrees (default: Costa Rica at -82.258937)
    """
    
    # Define observer location
    observer_location = EarthLocation(lat=latitude*u.deg, lon=longitude*u.deg, height=10*u.m)
    
    # Automatically determine timezone from coordinates
    tf = TimezoneFinder()
    timezone_str = tf.timezone_at(lat=latitude, lng=longitude)
    if timezone_str is None:
        print(f"Error: Could not determine timezone for coordinates ({latitude}, {longitude})")
        return None
    
    local_tz = pytz.timezone(timezone_str)
    
    # Parse input dates and times as LOCAL time
    try:
        start_datetime_local = local_tz.localize(datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M:%S"))
        end_datetime_local = local_tz.localize(datetime.strptime(f"{end_date} {end_time}", "%Y-%m-%d %H:%M:%S"))
    except ValueError as e:
        print(f"Error parsing date/time: {e}")
        print("Please use format YYYY-MM-DD for dates and HH:MM:SS for times")
        return None
    
    # Convert local time to UTC for astropy calculations
    start_datetime_utc = start_datetime_local.astimezone(pytz.utc)
    end_datetime_utc = end_datetime_local.astimezone(pytz.utc)
    
    # Create time array (one point every 10 minutes)
    time_step = timedelta(minutes=10)
    times_utc = []
    times_local_display = []
    current_time_utc = start_datetime_utc
    
    while current_time_utc <= end_datetime_utc:
        times_utc.append(current_time_utc)
        # Convert back to local time for display
        current_time_local = current_time_utc.astimezone(local_tz)
        times_local_display.append(current_time_local.strftime("%H:%M"))
        current_time_utc += time_step
    
    # Calculate sun and moon positions
    sun_altitudes = []
    sun_azimuths = []
    moon_altitudes = []
    moon_azimuths = []
    
    for t_utc in times_utc:
        # Convert to astropy Time object (UTC)
        obs_time = Time(t_utc, scale='utc')
        
        # Get sun position
        sun = get_body('sun', obs_time, observer_location)
        sun_altaz = sun.transform_to(AltAz(obstime=obs_time, location=observer_location))
        sun_altitudes.append(sun_altaz.alt.deg)
        sun_azimuths.append(sun_altaz.az.deg)
        
        # Get moon position
        moon = get_body('moon', obs_time, observer_location)
        moon_altaz = moon.transform_to(AltAz(obstime=obs_time, location=observer_location))
        moon_altitudes.append(moon_altaz.alt.deg)
        moon_azimuths.append(moon_altaz.az.deg)
    
    # Find rise/set times
    sunrise_time, sunset_time = find_sun_rise_set_times(start_datetime_local, end_datetime_local, observer_location, local_tz)
    moonrise_time, moonset_time = find_moon_rise_set_times(start_datetime_local, end_datetime_local, observer_location, local_tz)
    
    # Find twilight times
    civil_twilight_start, civil_twilight_end = find_twilight_times(start_datetime_local, end_datetime_local, observer_location, local_tz, -6)
    nautical_twilight_start, nautical_twilight_end = find_twilight_times(start_datetime_local, end_datetime_local, observer_location, local_tz, -12)
    astronomical_twilight_start, astronomical_twilight_end = find_twilight_times(start_datetime_local, end_datetime_local, observer_location, local_tz, -18)
    
    # Create figure with single plot
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot: Altitude vs Time for both sun and moon
    ax.plot(range(len(times_utc)), sun_altitudes, color='orange', linewidth=2, marker='o', markersize=4, label='Sun')
    ax.plot(range(len(times_utc)), moon_altitudes, color='lightblue', linewidth=2, marker='s', markersize=4, label='Moon')
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Horizon')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Altitude (degrees)', fontsize=12)
    ax.set_title('Sun and Moon Altitude vs Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    # Set x-axis limits to start and end indices
    ax.set_xlim(0, len(times_utc) - 1)
    
    # Set x-axis labels (every 6th point for readability)
    tick_positions = range(0, len(times_utc), 6)
    tick_labels = [times_local_display[i] for i in tick_positions]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("=" * 60)
    print("SUN AND MOON POSITION SUMMARY")
    print("=" * 60)
    print(f"Observation Period: {start_datetime_local.strftime('%Y-%m-%d %H:%M:%S %Z')} to {end_datetime_local.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"Observer Location: Latitude {latitude:.6f}°, Longitude {longitude:.6f}°")
    print(f"Timezone: {timezone_str}")
    print(f"\nSun Statistics:")
    print(f"  Maximum Altitude: {max(sun_altitudes):.2f}°")
    print(f"  Minimum Altitude: {min(sun_altitudes):.2f}°")
    print(f"  Time of Max Altitude (Solar Noon): {times_local_display[np.argmax(sun_altitudes)]} (Local Time)")
    print(f"  Azimuth Range: {min(sun_azimuths):.2f}° to {max(sun_azimuths):.2f}°")
    if sunrise_time:
        print(f"  Sunrise: {sunrise_time.strftime('%H:%M:%S %Z')}")
    if sunset_time:
        print(f"  Sunset: {sunset_time.strftime('%H:%M:%S %Z')}")
    print(f"\nTwilight Times:")
    if civil_twilight_start:
        print(f"  Civil Twilight Start: {civil_twilight_start.strftime('%H:%M:%S %Z')}")
    if civil_twilight_end:
        print(f"  Civil Twilight End: {civil_twilight_end.strftime('%H:%M:%S %Z')}")
    if nautical_twilight_start:
        print(f"  Nautical Twilight Start: {nautical_twilight_start.strftime('%H:%M:%S %Z')}")
    if nautical_twilight_end:
        print(f"  Nautical Twilight End: {nautical_twilight_end.strftime('%H:%M:%S %Z')}")
    if astronomical_twilight_start:
        print(f"  Astronomical Twilight Start: {astronomical_twilight_start.strftime('%H:%M:%S %Z')}")
    if astronomical_twilight_end:
        print(f"  Astronomical Twilight End: {astronomical_twilight_end.strftime('%H:%M:%S %Z')}")
    print(f"\nMoon Statistics:")
    print(f"  Maximum Altitude: {max(moon_altitudes):.2f}°")
    print(f"  Minimum Altitude: {min(moon_altitudes):.2f}°")
    print(f"  Time of Max Altitude: {times_local_display[np.argmax(moon_altitudes)]} (Local Time)")
    print(f"  Azimuth Range: {min(moon_azimuths):.2f}° to {max(moon_azimuths):.2f}°")
    if moonrise_time:
        print(f"  Moonrise: {moonrise_time.strftime('%H:%M:%S %Z')}")
    if moonset_time:
        print(f"  Moonset: {moonset_time.strftime('%H:%M:%S %Z')}")
    print(f"\n  (0°=North, 90°=East, 180°=South, 270°=West)")
    print("=" * 60)
    
    return fig


def find_sun_rise_set_times(start_datetime, end_datetime, observer_location, local_tz):
    """
    Find sunrise and sunset times within the given date range.
    
    Parameters:
    -----------
    start_datetime : datetime with timezone
        Start of observation period
    end_datetime : datetime with timezone
        End of observation period
    observer_location : EarthLocation
        Observer location
    local_tz : pytz timezone
        Local timezone
    
    Returns:
    --------
    sunrise_time : datetime or None
        Sunrise time in local timezone, or None if not found
    sunset_time : datetime or None
        Sunset time in local timezone, or None if not found
    """
    
    sunrise_time = None
    sunset_time = None
    
    # Search with 5-minute resolution
    time_step = timedelta(minutes=5)
    current_time_utc = start_datetime.astimezone(pytz.utc)
    end_time_utc = end_datetime.astimezone(pytz.utc)
    
    sun_altitude_prev = None
    
    while current_time_utc <= end_time_utc:
        obs_time = Time(current_time_utc, scale='utc')
        sun = get_body('sun', obs_time, observer_location)
        sun_altaz = sun.transform_to(AltAz(obstime=obs_time, location=observer_location))
        sun_altitude = sun_altaz.alt.deg
        
        # Check for crossing of horizon
        if sun_altitude_prev is not None:
            if sun_altitude_prev < 0 and sun_altitude >= 0 and sunrise_time is None:
                # Sunrise crossing
                sunrise_time = current_time_utc.replace(tzinfo=pytz.utc).astimezone(local_tz)
            elif sun_altitude_prev >= 0 and sun_altitude < 0 and sunset_time is None:
                # Sunset crossing
                sunset_time = current_time_utc.replace(tzinfo=pytz.utc).astimezone(local_tz)
        
        sun_altitude_prev = sun_altitude
        current_time_utc += time_step
    
    return sunrise_time, sunset_time


def find_moon_rise_set_times(start_datetime, end_datetime, observer_location, local_tz):
    """
    Find moonrise and moonset times within the given date range.
    
    Parameters:
    -----------
    start_datetime : datetime with timezone
        Start of observation period
    end_datetime : datetime with timezone
        End of observation period
    observer_location : EarthLocation
        Observer location
    local_tz : pytz timezone
        Local timezone
    
    Returns:
    --------
    moonrise_time : datetime or None
        Moonrise time in local timezone, or None if not found
    moonset_time : datetime or None
        Moonset time in local timezone, or None if not found
    """
    
    moonrise_time = None
    moonset_time = None
    
    # Search with 5-minute resolution
    time_step = timedelta(minutes=5)
    current_time_utc = start_datetime.astimezone(pytz.utc)
    end_time_utc = end_datetime.astimezone(pytz.utc)
    
    moon_altitude_prev = None
    
    while current_time_utc <= end_time_utc:
        obs_time = Time(current_time_utc, scale='utc')
        moon = get_body('moon', obs_time, observer_location)
        moon_altaz = moon.transform_to(AltAz(obstime=obs_time, location=observer_location))
        moon_altitude = moon_altaz.alt.deg
        
        # Check for crossing of horizon
        if moon_altitude_prev is not None:
            if moon_altitude_prev < 0 and moon_altitude >= 0 and moonrise_time is None:
                # Moonrise crossing
                moonrise_time = current_time_utc.replace(tzinfo=pytz.utc).astimezone(local_tz)
            elif moon_altitude_prev >= 0 and moon_altitude < 0 and moonset_time is None:
                # Moonset crossing
                moonset_time = current_time_utc.replace(tzinfo=pytz.utc).astimezone(local_tz)
        
        moon_altitude_prev = moon_altitude
        current_time_utc += time_step
    
    return moonrise_time, moonset_time


def find_twilight_times(start_datetime, end_datetime, observer_location, local_tz, altitude_threshold):
    """
    Find twilight times for a given sun altitude threshold.
    
    Parameters:
    -----------
    start_datetime : datetime with timezone
        Start of observation period
    end_datetime : datetime with timezone
        End of observation period
    observer_location : EarthLocation
        Observer location
    local_tz : pytz timezone
        Local timezone
    altitude_threshold : float
        Sun altitude threshold in degrees
        -6 for civil twilight, -12 for nautical, -18 for astronomical
    
    Returns:
    --------
    twilight_start : datetime or None
        Twilight start time in local timezone, or None if not found
    twilight_end : datetime or None
        Twilight end time in local timezone, or None if not found
    """
    
    twilight_start = None
    twilight_end = None
    
    # Search with 5-minute resolution
    time_step = timedelta(minutes=5)
    current_time_utc = start_datetime.astimezone(pytz.utc)
    end_time_utc = end_datetime.astimezone(pytz.utc)
    
    sun_altitude_prev = None
    
    while current_time_utc <= end_time_utc:
        obs_time = Time(current_time_utc, scale='utc')
        sun = get_body('sun', obs_time, observer_location)
        sun_altaz = sun.transform_to(AltAz(obstime=obs_time, location=observer_location))
        sun_altitude = sun_altaz.alt.deg
        
        # Check for crossing of twilight threshold
        if sun_altitude_prev is not None:
            if sun_altitude_prev < altitude_threshold and sun_altitude >= altitude_threshold and twilight_start is None:
                # Twilight start crossing (sun rising above threshold)
                twilight_start = current_time_utc.replace(tzinfo=pytz.utc).astimezone(local_tz)
            elif sun_altitude_prev >= altitude_threshold and sun_altitude < altitude_threshold and twilight_end is None:
                # Twilight end crossing (sun falling below threshold)
                twilight_end = current_time_utc.replace(tzinfo=pytz.utc).astimezone(local_tz)
        
        sun_altitude_prev = sun_altitude
        current_time_utc += time_step
    
    return twilight_start, twilight_end


# Example usage - Call this with your desired parameters
if __name__ == "__main__":
    # Default: Costa Rica, April 10-11, 2025
    plot_sun_moon_position(
        start_date='2025-04-10',
        start_time='18:11:00',
        end_date='2025-04-11',
        end_time='06:49:00',
        latitude=9.338212,
        longitude=-82.258937
    )