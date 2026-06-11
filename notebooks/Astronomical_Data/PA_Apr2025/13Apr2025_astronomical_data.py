import math
import numpy as np
import pytz
import pandas as pd
from datetime import datetime, timedelta
from astropy.coordinates import AltAz, EarthLocation, get_body, get_sun
from astropy.time import Time
import astropy.units as u
from timezonefinder import TimezoneFinder

print('✓ Imports OK')

# ── Configuration ─────────────────────────────────────────────────────────────
LAT      =  9.338212
LON      = -82.258937
START    = '2025-04-13 18:30'          # local time,  YYYY-MM-DD HH:MM
END      = '2025-04-14 06:30'          # local time,  YYYY-MM-DD HH:MM
INTERVAL = 1                           # minutes between rows
OUTPUT         = '04132025_PA_Astronomical_Data.csv'
WINDOWS_OUTPUT = '04132025_PA_Dark_Windows.csv'

ILLUMINANCE_THRESHOLD_LUX = 0.1411181818  # lux — rows at or below this are flagged as dark

# Timezone: auto-detected from coordinates, or set manually.
TIMEZONE = TimezoneFinder().timezone_at(lat=LAT, lng=LON)
# TIMEZONE = 'America/New_York'

print(f'Timezone  : {TIMEZONE}')
print(f'Location  : {LAT}°N, {LON}°E')
print(f'Range     : {START}  →  {END}  (every {INTERVAL} min)')
print(f'Threshold : {ILLUMINANCE_THRESHOLD_LUX} lux')

# ── Helfenstein & Veverka (1987) / Hapke (1986) Percent-Darkness Model ────────
# Disk-integrated parameters from H&V (1987) Table II:
#   w=0.21, h=0.07, S(0)=0.71, b=0.29, c=0.39, θ̄=20°
# Phase angle convention: 0° = full moon, 180° = new moon
# 0% darkness  = brightness at 2°   phase (smallest phase visible from Earth)
# 100% darkness = brightness at 179° phase

_w, _h, _S0, _b, _c, _tbar = 0.21, 0.07, 0.71, 0.29, 0.39, 20.0

def _hapke_brightness(alpha_deg):
    """Whole-disk brightness from Hapke (1986) using H&V (1987) Table II params."""
    alpha_deg = max(0.001, min(alpha_deg, 179.999))
    a = math.radians(alpha_deg)
    phase_fn   = 1 + _b * math.cos(a) + _c * (1.5 * math.cos(a) ** 2 - 0.5)
    opposition = _S0 / (1 + (1 / _h) * math.tan(a / 2))
    roughness  = 1 - 0.104 * math.sin(math.radians(_tbar)) * (1 - math.exp(-3.0 * a))
    t1 = (_w / 8) * ((1 + opposition) * phase_fn - 1)
    t2 = (2 * _w) / (3 * math.pi) * ((math.pi - a) * math.cos(a) + math.sin(a))
    return (t1 + t2) * roughness

_B_FULL = _hapke_brightness(2)    # 0%  darkness reference
_B_NEW  = _hapke_brightness(179)  # 100% darkness reference

def percent_darkness(alpha_deg):
    """Percent darkness from phase angle via H&V (1987) phase curve."""
    B = _hapke_brightness(alpha_deg)
    return round(min(100.0, max(0.0, 100 * (_B_FULL - B) / (_B_FULL - _B_NEW))), 2)

def _phase_angle_from_fraction(frac):
    """Phase angle (degrees) from illuminated fraction. 0° = full, 180° = new."""
    frac = max(0.0, min(1.0, frac))
    return 180.0 - math.degrees(math.acos(1.0 - 2.0 * frac))

print(f'H&V (1987) / Hapke (1986) model ready')
print(f'  Full moon reference  ( 2° phase) : {_B_FULL:.6f}')
print(f'  New moon reference   (179° phase): {_B_NEW:.6f}')

threshold_lux = float(ILLUMINANCE_THRESHOLD_LUX)
print(f'  Threshold lux      : {threshold_lux:.6f} lux')

# ── Illuminance & Phase Helpers ───────────────────────────────────────────────

def _sun_lux(alt):
    """Approximate horizontal illuminance (lux) from solar altitude (degrees)."""
    if alt >= 0:    return 133_775 * math.sin(math.radians(max(alt, 0))) ** 0.833
    if alt >= -6:   return max(0.0, 3.4   * (6  + alt))   # civil twilight
    if alt >= -12:  return max(0.0, 0.1   * (12 + alt))   # nautical twilight
    if alt >= -18:  return max(0.0, 0.001 * (18 + alt))   # astronomical twilight
    return 0.0

def _moon_lux(moon_alt, phase_angle_deg):
    if moon_alt <= 0:
        return 0.0
    hapke_scale = _hapke_brightness(phase_angle_deg) / _B_FULL
    return max(0.0, 0.27 * hapke_scale * math.sin(math.radians(moon_alt)))

def _phase_fraction(phase_angle_deg):
    """Illuminated fraction of the moon disk from phase angle."""
    return round((1 - math.cos(math.radians(180.0 - phase_angle_deg))) / 2, 4)

def _phase_name(frac):
    if frac <  0.01: return 'New Moon'
    if frac <  0.24: return 'Waxing Crescent'
    if frac <  0.26: return 'First Quarter'
    if frac <  0.49: return 'Waxing Gibbous'
    if frac <  0.51: return 'Half Moon'
    if frac <  0.74: return 'Waning Gibbous'
    if frac <  0.76: return 'Last Quarter'
    if frac <  0.99: return 'Waning Crescent'
    return 'Full Moon'

def _light_regime(sun_alt, moon_alt):
    if sun_alt >    0: return 'Daylit'
    if sun_alt >=  -6: return 'Civil Twilight'
    if sun_alt >= -12: return 'Nautical Twilight'
    if sun_alt >= -18: return 'Astronomical Twilight'
    if moon_alt >   0: return 'Moonlit'
    return 'Moonless Night'

print('✓ Helpers defined')

# ── Sunset / Sunrise Helper ───────────────────────────────────────────────────

def get_sun_event(date, lat, lon, tz_str, event='sunset'):
    """
    Find sunset or sunrise time for a given local date string ('YYYY-MM-DD').
    Samples solar altitude every minute over a ±15-hour window centred on local
    noon, finds the altitude zero-crossing, and linearly interpolates to
    sub-minute precision.
    Returns a local-time string 'HH:MM:SS', or 'N/A' if no crossing is found
    (e.g. polar day/night).
    event: 'sunset' or 'sunrise'
    """
    tz       = pytz.timezone(tz_str)
    location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg)

    local_noon = tz.localize(
        datetime.strptime(date, '%Y-%m-%d').replace(hour=12)
    )
    utc_noon = local_noon.astimezone(pytz.utc)

    # Sample every minute over a 30-hour window (±900 min) centred on local noon
    times_utc = [utc_noon + timedelta(minutes=m) for m in range(-900, 900)]
    obs_times = Time(times_utc)
    frame     = AltAz(obstime=obs_times, location=location)
    sun_alts  = get_sun(obs_times).transform_to(frame).alt.deg

    for i in range(len(sun_alts) - 1):
        rising  = sun_alts[i] < 0 and sun_alts[i + 1] >= 0   # sunrise crossing
        setting = sun_alts[i] >= 0 and sun_alts[i + 1] < 0   # sunset crossing

        if (event == 'sunrise' and rising) or (event == 'sunset' and setting):
            # Linear interpolation to sub-minute precision
            frac    = -sun_alts[i] / (sun_alts[i + 1] - sun_alts[i])
            precise = times_utc[i] + timedelta(minutes=frac)
            return precise.astimezone(tz).strftime('%H:%M:%S')

    return 'N/A'

print('✓ get_sun_event() defined')

# ── Data Generation ───────────────────────────────────────────────────────────

def generate_sky_data(lat, lon, start_local, end_local, interval_min, tz_str):
    tz       = pytz.timezone(tz_str)
    location = EarthLocation(lat=lat * u.deg, lon=lon * u.deg)
    rows     = []
    current  = start_local

    while current <= end_local:
        utc_dt   = tz.localize(current).astimezone(pytz.utc)
        obs_time = Time(utc_dt)
        frame    = AltAz(obstime=obs_time, location=location)

        sun      = get_sun(obs_time)
        sun_aa   = sun.transform_to(frame)
        moon_aa  = get_body('moon', obs_time, location).transform_to(frame)

        # Geocentric phase angle — avoids coordinate-transform warnings
        moon_geo  = get_body('moon', obs_time)
        phase_deg = 180.0 - sun.separation(moon_geo).deg
        pfrac     = _phase_fraction(phase_deg)

        rows.append({
            'date':                current.strftime('%Y-%m-%d'),
            'time':                current.strftime('%H:%M:%S'),
            'illuminance_lux': round(_sun_lux(sun_aa.alt.deg) + _moon_lux(moon_aa.alt.deg, phase_deg), 4),
            'light_regime':        _light_regime(sun_aa.alt.deg, moon_aa.alt.deg),
            'sun_altitude_deg':    round(sun_aa.alt.deg,  4),
            'sun_azimuth_deg':     round(sun_aa.az.deg,   4),
            'moon_altitude_deg':   round(moon_aa.alt.deg, 4),
            'moon_azimuth_deg':    round(moon_aa.az.deg,  4),
            'phase_angle_deg':     round(phase_deg, 4),
            'pct_darkness':        percent_darkness(phase_deg),
            'moon_phase_fraction': pfrac,
            'moon_phase_name':     _phase_name(pfrac),
        })
        current += timedelta(minutes=interval_min)

    return pd.DataFrame(rows)

print('✓ generate_sky_data() defined')

# ── Dark Windows ──────────────────────────────────────────────────────────────

def get_dark_windows(df, threshold_lux, lat, lon, tz_str):
    """
    Find contiguous time ranges where illuminance_lux <= threshold_lux.
    Prints a summary table and returns a DataFrame with one row per window:
      window_start, window_end, duration_min, threshold_lux, sunset, sunrise
    Sunset and sunrise are looked up for the window-start date and cached
    so each unique date is only computed once.
    """
    mask        = df['illuminance_lux'] <= threshold_lux
    transitions = mask.ne(mask.shift()).cumsum()
    groups      = df[mask].groupby(transitions[mask])

    print(f'\n{"─"*72}')
    print(f'  Threshold : {threshold_lux:.6f} lux')
    print(f'  Condition : illuminance_lux ≤ {threshold_lux:.6f} lux')
    print(f'{"─"*72}')

    if groups.ngroups == 0:
        print('  No windows found.')
        return pd.DataFrame(columns=[
            'window_start', 'window_end', 'duration_min',
            'threshold_lux', 'sunset', 'sunrise',
        ])

    # Cache sun events per date to avoid redundant astropy calls
    _sun_cache = {}

    def _cached(date, event):
        key = (date, event)
        if key not in _sun_cache:
            _sun_cache[key] = get_sun_event(date, lat, lon, tz_str, event)
        return _sun_cache[key]

    windows = []
    for _, group in groups:
        start    = group.iloc[0]['date']  + ' ' + group.iloc[0]['time']
        end      = group.iloc[-1]['date'] + ' ' + group.iloc[-1]['time']
        dur      = len(group)
        win_date = group.iloc[0]['date']   # use window-start date for sun events

        sunset  = _cached(win_date, 'sunset')
        sunrise = _cached(win_date, 'sunrise')

        print(f'  {start}  →  {end}  ({dur} min)  |  sunset {sunset}  sunrise {sunrise}')
        windows.append({
            'window_start': start,
            'window_end':   end,
            'duration_min': dur,
            'threshold_lux': round(threshold_lux, 8),
            'sunset':        sunset,
            'sunrise':       sunrise,
        })

    total = mask.sum()
    print(f'{"─"*72}')
    print(f'  Total : {total} min across {groups.ngroups} window(s)')

    return pd.DataFrame(windows)

print('✓ get_dark_windows() defined')

# ── Run ───────────────────────────────────────────────────────────────────────

fmt = '%Y-%m-%d %H:%M'
df = generate_sky_data(
    lat=LAT, lon=LON,
    start_local=datetime.strptime(START, fmt),
    end_local=datetime.strptime(END,   fmt),
    interval_min=INTERVAL,
    tz_str=TIMEZONE,
)

# Tag each row and save main CSV
col = 'below_threshold'
df[col] = df['illuminance_lux'] <= threshold_lux
df.to_csv(OUTPUT, index=False)
print(f'✅ Saved {len(df)} rows → {OUTPUT}')

# Find dark windows, print summary, and save to CSV
windows_df = get_dark_windows(df, threshold_lux, LAT, LON, TIMEZONE)
windows_df.to_csv(WINDOWS_OUTPUT, index=False)
print(f'\n✅ Saved {len(windows_df)} window(s) → {WINDOWS_OUTPUT}')

# ── Data Preview ──────────────────────────────────────────────────────────────

print('\n── Main data (first 10 rows) ──')
print(df.head(10).to_string(index=False))

print('\n── Dark windows ──')
print(windows_df.to_string(index=False))
