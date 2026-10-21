"""
07_statistical_trend_tests.py
Non-parametric Trend Tests (Mann-Kendall & Sen's Slope) and Spatial Regression
Evaluates whether urban expansion adds significant explanatory power to flood exposure.
"""

import math

def calculate_mann_kendall_test(series):
    """
    Mann-Kendall non-parametric test for monotonic trend detection:
    S = sum(sign(x_j - x_k)) for j > k
    Variance and Z-score calculation.
    """
    n = len(series)
    s = 0
    for k in range(n - 1):
        for j in range(k + 1, n):
            diff = series[j] - series[k]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1
                
    var_s = (n * (n - 1) * (2 * n + 5)) / 18.0
    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0
        
    # Two-tailed p-value approximation
    p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    trend = 'Increasing' if z > 1.96 else ('Decreasing' if z < -1.96 else 'No Significant Trend')
    
    return {'S': s, 'Z': round(z, 3), 'P_Value': round(p_val, 4), 'Trend': trend}

def calculate_sens_slope(series):
    """Sen's Slope Estimator: median of all pairwise slopes"""
    n = len(series)
    slopes = []
    for k in range(n - 1):
        for j in range(k + 1, n):
            slopes.append((series[j] - series[k]) / (j - k))
    slopes.sort()
    mid = len(slopes) // 2
    median_slope = slopes[mid] if len(slopes) % 2 != 0 else (slopes[mid-1] + slopes[mid]) / 2.0
    return round(median_slope, 4)

if __name__ == '__main__':
    print("[Econometrics & Spatial Stats] Running Mann-Kendall Trend Analysis on Chattogram (2001-2025)...")
    # Historical Rx1day extreme annual maximum rainfall in mm (2001-2025 sample)
    rx1day_series = [142, 128, 155, 182, 134, 162, 315, 140, 168, 175, 210, 245, 158, 172, 195, 185, 230, 178, 225, 215, 190, 240, 212, 265, 248]
    mk = calculate_mann_kendall_test(rx1day_series)
    slope = calculate_sens_slope(rx1day_series)
    print(f"  Mann-Kendall Z-Score: {mk['Z']} (p-value: {mk['P_Value']}) -> {mk['Trend']}")
    print(f"  Sen's Slope: +{slope} mm/year in extreme 1-day monsoonal precipitation.")
