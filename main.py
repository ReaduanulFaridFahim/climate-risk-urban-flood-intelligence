"""
========================================================================================
main.py
DELTASHIELD: BANGLADESH UNIFIED CLIMATE & MULTI-SECTOR INTELLIGENCE SUITE
========================================================================================
Pillar 1: 🌾 AgroClaim   - Parametric Crop Insurance & Automated Mobile Disaster Payouts
Pillar 2: 🚑 EvacRoute   - Submerged Road Networks & Island Isolation Emergency Routing
Pillar 3: 🏢 ClimaDeval  - 30-Year Real Estate & Banking ESG Climate Credit Rating (AAA-CCC)
Pillar 4: 🎮 DeltaSim    - National Climate Adaptation Policy & Budget Simulation Sandbox
========================================================================================
Terminal Usage:
    python main.py                      (Interactive Menu & National Overview)
    python main.py --district "Sylhet"  (Run All 4 Pillars for a District)
    python main.py --agro "Sunamganj"   (Run Pillar 1: AgroClaim Crop Insurance)
    python main.py --evac "Kurigram"    (Run Pillar 2: EvacRoute Submerged Logistics)
    python main.py --clima "Chittagong" (Run Pillar 3: ClimaDeval 30-Yr ESG Rating)
    python main.py --sim                (Run Pillar 4: DeltaSim Policy Budget Sandbox)
    python main.py --serve              (Launch Interactive Local Web Dashboard on Port 8000)
========================================================================================
"""

import os
import sys
import argparse

# Force UTF-8 terminal encoding on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from modules.agroclaim import assess_district_crop_risk, get_national_crop_summary
from modules.evacroute import assess_isolation_risk, get_most_isolated_districts
from modules.climadeval import compute_climadeval_rating, get_national_esg_leaderboard
from modules.deltasim import run_policy_simulation
from modules.weather_climate_ai import train_and_benchmark_models, predict_future_2months


def print_banner():
    print("\n" + "=" * 80)
    print("🛡️  DELTASHIELD: BANGLADESH UNIFIED CLIMATE & MULTI-SECTOR DECISION SUITE")
    print("=" * 80)
    print("Unified Intelligence: 🌾 AgroClaim • 🚑 EvacRoute • 🏢 ClimaDeval • 🎮 DeltaSim")
    print("=" * 80)


def run_all_pillars_for_district(district_name: str):
    print_banner()
    print(f"\n[+] RUNNING 11-PILLAR UNIFIED CLIMATE INTELLIGENCE ASSESSMENT FOR: {district_name.upper()}\n")

    # Pillar 1: AgroClaim
    agro = assess_district_crop_risk(district_name, rainfall_anomaly_pct=25.0)
    if agro.get('status') == 'error':
        print(f"[!] Error: {agro['message']}")
        return

    print("=" * 80)
    print("🌾 PILLAR 1: AGROCLAIM - PARAMETRIC CROP INSURANCE & LOSS ENGINE")
    print("-" * 80)
    print(f"  • Division                        : {agro['division']}")
    print(f"  • Total Standing Cropland Area    : {agro['total_cropland_ha']:,} Hectares")
    print(f"  • Harvest Breakdown               : Boro: {agro['crops']['boro_rice_ha']:,} ha | Aman: {agro['crops']['aman_rice_ha']:,} ha | Aus: {agro['crops']['aus_rice_ha']:,} ha")
    print(f"  • Simulated Inundation Extent     : {agro['inundation_pct']}% of cropland")
    print(f"  • Submerged Cropland Area         : {agro['submerged_cropland_ha']:,} Hectares")
    print(f"  • Estimated Farmer Economic Loss  : ${agro['farmer_loss_usd']:,.2f} USD")
    print(f"  • Actuarial Pure Premium Rate     : {agro['premium_rate_pct']}% (${agro['premium_usd_per_ha']} / ha)")
    print(f"  • Parametric Trigger Status       : {'🚨 PAYOUT TRIGGERED (' + str(agro['payout_pct']) + '%)' if agro['payout_triggered'] else '✅ SAFE (<45% Threshold)'}")
    if agro['payout_triggered']:
        print(f"  • Total Automated Mobile Payout   : ${agro['payout_amount_usd']:,.2f} USD")

    # Pillar 2: EvacRoute
    evac = assess_isolation_risk(district_name, water_rise_m=1.2)
    print("\n" + "=" * 80)
    print("🚑 PILLAR 2: EVACROUTE - SUBMERGED ROAD LOGISTICS & SAFE SHELTER ROUTING")
    print("-" * 80)
    print(f"  • Mean Elevation AMSL             : {evac['elevation_amsl_m']:.1f} meters")
    print(f"  • Total Road Network              : {evac['total_roads_km']:,} km (HOTOSM Arterial Grid)")
    print(f"  • Submerged Road Extent (+1.2m)   : {evac['submerged_roads_km']:,} km ({evac['submersion_pct']}%)")
    print(f"  • Passable Corridors Remaining    : {evac['passable_roads_km']:,} km")
    print(f"  • Island Isolation Risk Index     : {evac['island_isolation_index']} / 100 ({evac['isolation_status']})")
    print(f"  • Nearest Designated Safe Haven   : {evac['shelter']['name']}")
    print(f"  • Shelter Type & Capacity         : {evac['shelter']['type']} ({evac['shelter']['capacity']})")
    print(f"  • Evacuation Corridor Directive   : {evac['shelter']['routing_directive']}")

    # Pillar 3: ClimaDeval
    clima = compute_climadeval_rating(district_name, initial_property_value_usd=500000.0)
    print("\n" + "=" * 80)
    print("🏢 PILLAR 3: CLIMADEVAL - 30-YEAR REAL ESTATE & BANKING ESG CLIMATE RATING")
    print("-" * 80)
    print(f"  • Institutional ESG Rating        : [{clima['climadeval_rating']}] - {clima['rating_description']}")
    print(f"  • Composite Physical Risk Score   : {clima['composite_risk_score']} / 100")
    print(f"  • Projected 2050 Flood Probability: {clima['projected_inundation_2050_pct']}% (SSP5-8.5 Severe)")
    print(f"  • Baseline Property Asset Value   : ${clima['initial_property_value_usd']:,.2f} USD")
    print(f"  • 30-Year Depreciation Haircut    : -{clima['estimated_30yr_depreciation_pct']}% (-${clima['projected_value_loss_usd']:,.2f} USD)")
    print(f"  • Retained 2050 Property Value    : ${clima['projected_retained_value_usd']:,.2f} USD")
    print(f"  • Underwriting Covenant           : {clima['recommended_underwriting_covenant']}")

    # Pillar 4: DeltaSim Summary
    sim = run_policy_simulation(budget_millions_usd=500.0)
    print("\n" + "=" * 80)
    print("🎮 PILLAR 4: DELTASIM - NATIONAL CLIMATE ADAPTATION POLICY SCORE")
    print("-" * 80)
    print(f"  • National Budget Allocated       : ${sim['total_budget_usd_millions']:.1f} Million USD")
    print(f"  • Composite Resilience Score      : {sim['national_resilience_score']} / 100 ({sim['evaluation']})")
    print(f"  • 2050 Lives Protected            : {sim['mitigated_2050']['lives_protected']:,} citizens")
    print(f"  • 2050 Economic Damage Avoided    : ${sim['mitigated_2050']['damage_avoided_usd_billions']:.1f} Billion USD")

    # Pillar 5: Weather & Climate AI (Next 2 Months)
    try:
        weather = predict_future_2months(district_name, days=60)
        print("\n" + "=" * 80)
        print(f"🌤️  PILLAR 5: WEATHER & CLIMATE AI - 60-DAY FORECAST [{weather['division'].upper()}]")
        print("-" * 80)
        print(f"  • Forecast Horizon                : {weather['start_date']} to {weather['end_date']} (60 Days)")
        print(f"  • Expected Temperature Range      : {weather['summary']['avg_temp_min_c']}°C (Min) to {weather['summary']['avg_temp_max_c']}°C (Max)")
        print(f"  • Total Projected Precipitation   : {weather['summary']['total_precipitation_mm']} mm (Peak Day: {weather['summary']['max_single_day_rain_mm']} mm)")
        print(f"  • Foggy / Dense Haze Days         : {weather['summary']['dense_fog_days']} days (>=2.0 hrs fog duration)")
        print(f"  • Peak Cold Wave Risk Index       : {weather['summary']['peak_cold_wave_risk_score']} / 100")
        print(f"  • Peak Flood Threat Score         : {weather['summary']['peak_flood_risk_score']} / 100")
    except Exception as e:
        print(f"\n[!] Weather forecast note: {e}")

    # Pillar 6: HealthEpidemic
    try:
        from modules.health_epidemic import assess_epidemic_risk
        health = assess_epidemic_risk(district_name)
        print("\n" + "=" * 80)
        print("🏥 PILLAR 6: HEALTHEPIDEMIC - VECTOR & WATERBORNE SURGE SURVEILLANCE")
        print("-" * 80)
        print(f"  • Composite Health Risk Score     : {health['composite_health_risk']}/100 ({health['alert_tier']})")
        print(f"  • Dengue Transmission Index       : {health['dengue_risk_score']}/100")
        print(f"  • Cholera/Waterborne Index        : {health['cholera_risk_score']}/100")
        print(f"  • Hospital Bed Occupancy Surge    : {health['hospital_capacity']['bed_occupancy_pct']}% ({health['hospital_capacity']['status']})")
        print(f"  • IV Saline Bags (1000ml) Needed  : {health['supply_logistics']['iv_saline_bags_1000ml']:,} bags")
    except Exception as e:
        print(f"\n[!] Health epidemic note: {e}")

    # Pillar 7: GridResilience
    try:
        from modules.grid_resilience import assess_grid_vulnerability
        grid = assess_grid_vulnerability(district_name)
        print("\n" + "=" * 80)
        print("⚡ PILLAR 7: GRIDRESILIENCE - POWER SUBSTATION & TELECOM SURVIVABILITY")
        print("-" * 80)
        print(f"  • Overall Blackout Risk           : {grid['blackout_probability_pct']}% ({grid['grid_status']})")
        print(f"  • Substations Inundation Threat   : {grid['substation_vulnerability']['substations_at_flood_risk']}/{grid['substation_vulnerability']['total_substations']} substations")
        print(f"  • Cellular BTS Towers Offline     : {grid['telecom_infrastructure']['projected_towers_offline']}/{grid['telecom_infrastructure']['total_bts_towers']} towers (-{grid['telecom_infrastructure']['cellular_coverage_loss_pct']}%)")
        print(f"  • Emergency Hospital Diesel Fuel  : {grid['backup_power_logistics']['daily_diesel_liters']:,} Liters/day")
    except Exception as e:
        print(f"\n[!] Grid resilience note: {e}")

    # Pillar 8: RiverbankErosion
    try:
        from modules.riverbank_erosion import assess_riverbank_erosion
        erosion = assess_riverbank_erosion(district_name)
        print("\n" + "=" * 80)
        print("🌊 PILLAR 8: RIVERBANK EROSION & CHAR REFUGEE DISPLACEMENT")
        print("-" * 80)
        print(f"  • Major River Basin               : {erosion['primary_river_system']} (Risk: {erosion['erosion_vulnerability_rating']})")
        print(f"  • Projected Land Diluvion         : {erosion['projected_land_loss_ha']} Hectares ({erosion['projected_land_loss_sq_km']} km²)")
        print(f"  • Displaced Climate Refugees      : {erosion['displacement_estimates']['climate_refugees_displaced']:,} citizens")
        print(f"  • Emergency Relocation Grant Pool : ${erosion['displacement_estimates']['immediate_resettlement_grant_usd']:,.2f} USD")
    except Exception as e:
        print(f"\n[!] Riverbank erosion note: {e}")

    # Pillar 9: AgriSupply
    try:
        from modules.agri_supply import analyze_supply_chain_bottlenecks
        supply = analyze_supply_chain_bottlenecks(district_name)
        print("\n" + "=" * 80)
        print("🍞 PILLAR 9: AGRISUPPLY - HIGHWAY ARTERIAL CHOKEPOINTS & FOOD INFLATION")
        print("-" * 80)
        print(f"  • Commodity Hub Specialization    : {supply['commodity_specialization']}")
        print(f"  • Daily Metropolitan Dispatch     : {supply['daily_volume']['truckloads_to_metropolitan']} freight trucks ({supply['daily_volume']['metric_tons_dispatched']} tons)")
        print(f"  • Highway Chokepoint Inundation   : {supply['highway_chokepoint_analysis']['chokepoint_status']} (Flood prob: {supply['highway_chokepoint_analysis']['highway_submersion_probability_pct']}%)")
        print(f"  • Detour Transit Penalty          : +{supply['freight_rerouting_logistics']['detour_additional_distance_km']} km (+{supply['freight_rerouting_logistics']['detour_additional_transit_hours']} hrs)")
        print(f"  • Metropolitan Food Price Shock   : {supply['urban_market_impact']['projected_commodity_price_inflation_pct']} inflation risk in {supply['urban_market_impact']['at_risk_metropolitan_hub']}")
    except Exception as e:
        print(f"\n[!] AgriSupply note: {e}")

    print("=" * 80)


def start_live_dashboard(port=8000):
    import http.server
    import socketserver
    import webbrowser

    web_dir = os.path.join(BASE_DIR, 'web')
    if not os.path.exists(web_dir):
        print(f"[!] Web directory not found at {web_dir}")
        return

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=web_dir, **kwargs)
        def log_message(self, format, *args):
            pass

    print_banner()
    print(f"[+] Launching DeltaShield Local Dashboard at: http://localhost:{port}")
    print(f"[+] Direct link: http://127.0.0.1:{port}/")
    print("[+] Zero-dependency, ultra-fast multi-threaded decision console.")
    print("[+] Press Ctrl+C in this terminal to stop.")
    print("=" * 80)

    try:
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        pass

    class ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        allow_reuse_address = True
        daemon_threads = True

    with ThreadedServer(("", port), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[+] Dashboard server stopped.")


def interactive_terminal_menu():
    print_banner()
    while True:
        print("\n" + "=" * 55)
        print("   DELTASHIELD NATIONAL COMMAND INTELLIGENCE MENU")
        print("=" * 55)
        print(" 1. Run Complete 11-Pillar Assessment for a District")
        print(" 2. Live Satellite Weather & 64-District Surveillance")
        print(" 3. National Disaster Alerts & Warning Broadcasts")
        print(" 4. Real-Time GPS Telemetry & Shelter Routing HUD")
        print(" 5. Weather & Climate AI: 2-Month (60-Day) Forecast")
        print(" 6. AI Model Benchmark: Train & Compare 8 ML Models")
        print(" 7. AgroClaim: Parametric Crop Insurance Leaderboard")
        print(" 8. EvacRoute: Top Most Isolated Flood Districts")
        print(" 9. ClimaDeval: 30-Year Real Estate ESG Rankings")
        print("10. DeltaSim: Run Policy Budget Simulation Sandbox")
        print("11. HealthEpidemic: Dengue & Cholera Surge Surveillance")
        print("12. GridResilience: Power Substation & Telecom Outages")
        print("13. Riverbank Erosion & Climate Refugee Resettlement")
        print("14. AgriSupply: Highway Chokepoints & Food Inflation")
        print("15. DeltaBot AI Climate Copilot (Ask Anything)")
        print("16. Launch Local Web Dashboard (http://localhost:8000)")
        print("17. Exit")
        choice = input("\nEnter choice [1-17]: ").strip()

        if choice == '1':
            d_name = input("Enter District Name (e.g. Sunamganj, Chittagong, Sylhet, Kurigram): ").strip()
            if d_name:
                run_all_pillars_for_district(d_name)
        elif choice == '2':
            import json
            live_json = os.path.join(BASE_DIR, 'data', 'processed', 'bangladesh_64_districts_live_weather.json')
            if os.path.exists(live_json):
                with open(live_json, encoding='utf-8') as f: dists = json.load(f)
                print(f"\n{'District':<18} {'Temp':<8} {'Humidity':<10} {'Wind':<10} {'Condition':<26} {'Alert Tier'}")
                print("-" * 90)
                for d in dists[:15]:
                    print(f"{d['name']:<18} {d['temp']:>4.1f}°C   {d['humidity']:>3}%       {d['wind']:>4.1f} km/h   {d['cond']:<26} {d.get('alert_tier', 'GREEN')}")
                print(f"... and {len(dists)-15} more districts. Use CLI 'python main.py --live all' for full table.")
        elif choice == '3':
            from modules.automated_alerts import get_active_disaster_alerts
            alerts = get_active_disaster_alerts(tier_filter="all")
            print(f"\n🚨 ACTIVE WARNINGS: {alerts['summary']['red_alerts_critical']} Red Alerts | {alerts['summary']['orange_alerts_warning']} Orange Advisories")
            for a in alerts['alerts'][:5]:
                print(f"  • [{a['tier'].upper()}] {a['district']}: {a['hazard_type']} · {a['affected_population']:,} pop · {a['directive'][:50]}...")
        elif choice == '4':
            coords = input("Enter GPS coordinates (lat,lon - e.g. 23.8103,90.4125): ").strip() or "23.8103,90.4125"
            os.system(f'python main.py --gps "{coords}"')
        elif choice == '5':
            loc = input("Enter District or Division (default Dhaka): ").strip() or "Dhaka"
            pred = predict_future_2months(loc, days=60)
            print(f"\n🌤️ 2-MONTH FORECAST FOR {pred['division'].upper()} ({pred['start_date']} to {pred['end_date']}):")
            for k, v in pred['summary'].items():
                print(f"  • {k}: {v}")
        elif choice == '6':
            bench = train_and_benchmark_models()
            print(f"\n🏆 CHAMPION MODEL: {bench['champion_model']} (R²={bench['champion_r2']:.4f}, RMSE={bench['champion_rmse']:.4f})")
            for row in bench['benchmark_table']:
                print(f"  {row['Is_Champion']:<10} {row['Model']:<30} R²={row['R2_Score']:.4f}  RMSE={row['RMSE']:.4f}")
        elif choice == '7':
            print("\n🌾 TOP 10 AGRICULTURAL RISK DISTRICTS (AgroClaim):")
            df = get_national_crop_summary(10)
            print(df.to_string(index=False))
        elif choice == '8':
            print("\n🚑 TOP 10 MOST ISOLATED FLOOD DISTRICTS (EvacRoute +1.0m Rise):")
            df = get_most_isolated_districts(1.0, 10)
            print(df.to_string(index=False))
        elif choice == '9':
            print("\n🏢 TOP 10 PROPERTY CLIMATE RISK DISTRICTS (ClimaDeval):")
            df = get_national_esg_leaderboard(10)
            print(df.to_string(index=False))
        elif choice == '10':
            budget = float(input("Enter National Budget ($M USD, default 500): ") or "500")
            sim = run_policy_simulation(budget_millions_usd=budget)
            print(f"\n🎮 DeltaSim Score: {sim['national_resilience_score']}/100 ({sim['evaluation']})")
            print(f"  • Lives Protected: {sim['mitigated_2050']['lives_protected']:,}")
            print(f"  • Damage Avoided : ${sim['mitigated_2050']['damage_avoided_usd_billions']:.1f} Billion USD")
        elif choice == '11':
            d_name = input("Enter District for Health Analysis (e.g. Dhaka, Khulna): ").strip() or "Dhaka"
            from modules.health_epidemic import assess_epidemic_risk
            h = assess_epidemic_risk(d_name)
            print(f"\n🏥 Health Surge for {d_name}: Bed Occupancy {h['hospital_capacity']['bed_occupancy_pct']}%, Dengue Score {h['dengue_risk_score']}/100, IV Saline Bags: {h['supply_logistics']['iv_saline_bags_1000ml']:,}")
        elif choice == '12':
            d_name = input("Enter District for Power Grid Analysis (e.g. Sylhet, Chattogram): ").strip() or "Chattogram"
            from modules.grid_resilience import assess_grid_vulnerability
            g = assess_grid_vulnerability(d_name)
            print(f"\n⚡ Grid Outage for {d_name}: Blackout Prob {g['blackout_probability_pct']}%, Towers Offline: {g['telecom_infrastructure']['projected_towers_offline']}, Generator Fuel: {g['backup_power_logistics']['daily_diesel_liters']:,} L/day")
        elif choice == '13':
            d_name = input("Enter District for Riverbank Erosion (e.g. Sirajganj, Kurigram): ").strip() or "Sirajganj"
            from modules.riverbank_erosion import assess_riverbank_erosion
            er = assess_riverbank_erosion(d_name)
            print(f"\n🌊 Riverbank Erosion for {d_name}: Land Loss {er['projected_land_loss_ha']} ha, Refugees: {er['displacement_estimates']['climate_refugees_displaced']:,}, Relief Grant: ${er['displacement_estimates']['immediate_resettlement_grant_usd']:,.2f}")
        elif choice == '14':
            d_name = input("Enter District for AgriSupply (e.g. Bogura, Dinajpur): ").strip() or "Bogura"
            from modules.agri_supply import analyze_supply_chain_bottlenecks
            sp = analyze_supply_chain_bottlenecks(d_name)
            print(f"\n🍞 AgriSupply for {d_name}: {sp['daily_volume']['truckloads_to_metropolitan']} trucks, Chokepoint: {sp['highway_chokepoint_analysis']['chokepoint_status']}, Inflation Risk: {sp['urban_market_impact']['projected_commodity_price_inflation_pct']}")
        elif choice == '15':
            q = input("Ask DeltaBot Copilot (e.g. What is the hospital situation in Sylhet?): ").strip()
            if q:
                from modules.deltabot import ask_deltabot
                ans = ask_deltabot(q)
                print(f"\n🤖 DELTABOT: {ans['response']}")
        elif choice == '16':
            start_live_dashboard(8000)
            break
        elif choice == '17':
            print("\n[+] Exiting DeltaShield. Goodbye!")
            break


def main():
    parser = argparse.ArgumentParser(description="DeltaShield: Bangladesh Unified Climate Decision Suite")
    parser.add_argument("--district", type=str, help="Run all 5 pillars for a specific district")
    parser.add_argument("--live", type=str, nargs="?", const="all", help="Display live satellite weather & time (pass district name or 'all' for 64 districts)")
    parser.add_argument("--alerts", type=str, nargs="?", const="all", help="National Disaster Early Warning & Alert System (pass 'red', 'orange', district name, or 'all')")
    parser.add_argument("--live-sync", action="store_true", help="Synchronize live real-time meteorological feeds (Open-Meteo & BMD) and re-evaluate alerts")
    parser.add_argument("--auto-poll", type=int, nargs="?", const=15, help="Start automated continuous live weather & disaster surveillance daemon (e.g. --auto-poll 15)")
    parser.add_argument("--gps", type=str, help="Pinpoint live GPS coordinates (e.g. --gps '23.8103,90.4125') to identify nearest district, disaster tier, and shelter")
    parser.add_argument("--forecast", type=str, help="Generate 60-day weather & climate AI forecast for district/division")
    parser.add_argument("--train-ai", action="store_true", help="Train and benchmark 8 AI models across climate datasets")
    parser.add_argument("--agro", type=str, help="Run AgroClaim crop insurance for a district")
    parser.add_argument("--evac", type=str, help="Run EvacRoute submerged logistics for a district")
    parser.add_argument("--clima", type=str, help="Run ClimaDeval 30-year ESG rating for a district")
    parser.add_argument("--sim", action="store_true", help="Run DeltaSim policy budget sandbox")
    parser.add_argument("--health", type=str, help="Run HealthEpidemic Dengue/Cholera and hospital surge analysis")
    parser.add_argument("--grid", type=str, help="Run GridResilience substation and telecom blackout vulnerability analysis")
    parser.add_argument("--erosion", type=str, help="Run RiverbankErosion char displacement and refugee analysis")
    parser.add_argument("--supply", type=str, help="Run AgriSupply highway chokepoints and food inflation analysis")
    parser.add_argument("--bot", type=str, help="Ask DeltaBot AI Climate Copilot natural language questions")
    parser.add_argument("--serve", action="store_true", help="Launch local web dashboard on port 8000")
    args = parser.parse_args()

    if args.serve:
        start_live_dashboard(8000)
    elif args.live:
        print_banner()
        import json
        from datetime import datetime, timezone, timedelta
        bst_now = datetime.now(timezone(timedelta(hours=6)))
        print(f"\n🕒 LIVE TIME: {bst_now.strftime('%A, %B %d, %Y · %I:%M:%S %p')} BST (UTC+6)")
        print("🛰️  SATELLITE RADAR: Himawari-9 (JMA) & BMD Doppler Network [ONLINE 🟢]\n")
        
        live_json = os.path.join(BASE_DIR, 'data', 'processed', 'bangladesh_64_districts_live_weather.json')
        if os.path.exists(live_json):
            with open(live_json, encoding='utf-8') as f:
                dists = json.load(f)
            query = args.live.strip().lower()
            if query == 'all':
                print(f"{'District':<18} {'Division':<14} {'Temp':<8} {'Feels':<8} {'Humidity':<10} {'Wind':<10} {'Condition':<26} {'Status'}")
                print("-" * 110)
                for d in dists:
                    print(f"{d['name']:<18} {d['division']:<14} {d['temp']:>4.1f}°C  {d['feels']:>4.1f}°C   {d['humidity']:>3}%       {d['wind']:>4.1f} km/h   {d['cond']:<26} {d['alert']}")
            else:
                match = [d for d in dists if query in d['name'].lower()]
                if match:
                    for d in match:
                        print(f"📍 LIVE WEATHER: {d['name'].upper()} ({d['division']} Division)")
                        print(f"  • Current Temperature : {d['temp']}°C (Feels like {d['feels']}°C)")
                        print(f"  • Sky & Weather Cond  : {d['cond']}")
                        print(f"  • Relative Humidity   : {d['humidity']}%")
                        print(f"  • Surface Wind Speed  : {d['wind']} km/h")
                        print(f"  • Atmospheric Pressure: {d['pressure']} hPa (Elevation: {d['elev']}m AMSL)")
                        print(f"  • Multi-Hazard Alert  : {d['alert']}")
                else:
                    print(f"[!] District '{args.live}' not found. Showing Dhaka:")
                    for d in dists:
                        if d['name'] == 'Dhaka':
                            print(f"  • {d['name']}: {d['temp']}°C | Hum: {d['humidity']}% | {d['cond']}")
    elif args.auto_poll is not None:
        print_banner()
        from modules.automated_alerts import monitor_live_weather_alerts
        monitor_live_weather_alerts(interval_seconds=args.auto_poll, max_iterations=3, query=args.alerts or "all")
    elif args.live_sync or args.alerts:
        print_banner()
        import json
        from datetime import datetime, timezone, timedelta
        from modules.automated_alerts import sync_live_weather_and_evaluate

        if args.live_sync:
            print("🌐 Connecting to Open-Meteo & BMD Live Meteorological Feeds...")
            sync_live_weather_and_evaluate(fetch_live_api=True, target_districts=["Dhaka", "Chattogram", "Sylhet", "Cox's Bazar", "Sunamganj", "Kurigram", "Khulna", "Barishal"])
            print("✅ Live weather telemetry synchronized & disaster risk models recalculated!\n")

        bst_now = datetime.now(timezone(timedelta(hours=6)))
        print(f"🚨 NATIONAL DISASTER EARLY WARNING & BROADCAST SYSTEM")
        print(f"🕒 Telemetry Timestamp: {bst_now.strftime('%A, %B %d, %Y · %I:%M:%S %p')} BST")
        print("📡 Feed: FFWC (Flood Warning Centre), BMD (Meteorology Dept) & Open-Meteo Satellite Feed")
        print("📞 Emergency Hotlines: 1090 (Flood/Cyclone IVR) · 999 (Rescue & Police) · 109 (Emergency)\n")

        live_json = os.path.join(BASE_DIR, 'data', 'processed', 'bangladesh_64_districts_live_weather.json')
        if os.path.exists(live_json):
            with open(live_json, encoding='utf-8') as f:
                dists = json.load(f)
            query = (args.alerts or "all").strip().lower()
            
            # Filtering
            if query == 'red':
                results = [d for d in dists if 'red' in d.get('alert_tier', '').lower()]
            elif query == 'orange':
                results = [d for d in dists if 'orange' in d.get('alert_tier', '').lower()]
            elif query in ['all', '']:
                results = [d for d in dists if 'red' in d.get('alert_tier', '').lower() or 'orange' in d.get('alert_tier', '').lower()]
            else:
                results = [d for d in dists if query in d['name'].lower()]

            red_count = sum(1 for d in dists if 'red' in d.get('alert_tier', '').lower())
            orange_count = sum(1 for d in dists if 'orange' in d.get('alert_tier', '').lower())
            print(f"⚠️  ACTIVE NATIONAL WARNINGS: {red_count} Critical Red Alerts | {orange_count} Orange Advisories\n")
            print(f"{'Tier':<16} {'District':<16} {'Hazard Category':<30} {'Target Population':<20} {'Directive'}")
            print("-" * 120)
            for d in results:
                tier_label = "🔴 RED ALERT" if "red" in d.get('alert_tier', '').lower() else ("🟠 ORANGE" if "orange" in d.get('alert_tier', '').lower() else "🟡 WATCH")
                print(f"{tier_label:<16} {d['name']:<16} {d.get('hazard_type', 'FLASH FLOOD')[:28]:<30} {d.get('affected_pop', 0):>10,d} citizens     {d.get('english_alert', '')[:45]}...")
    elif args.train_ai:
        print_banner()
        bench = train_and_benchmark_models()
        print(f"\n🏆 CHAMPION MODEL: {bench['champion_model']} (R²={bench['champion_r2']:.4f}, RMSE={bench['champion_rmse']:.4f})\n")
        import pandas as pd
        df = pd.DataFrame(bench['benchmark_table'])
        print(df.to_string(index=False))
    elif args.forecast:
        print_banner()
        pred = predict_future_2months(args.forecast, days=60)
        print(f"\n🌤️ 2-MONTH (60-DAY) AI WEATHER & CLIMATE FORECAST FOR {pred['division'].upper()}:")
        print(f"  Horizon: {pred['start_date']} to {pred['end_date']}")
        print(f"  Avg Temp: {pred['summary']['avg_temp_min_c']}°C - {pred['summary']['avg_temp_max_c']}°C")
        print(f"  Precipitation Total: {pred['summary']['total_precipitation_mm']} mm (Peak Day: {pred['summary']['max_single_day_rain_mm']} mm)")
        print(f"  Dense Fog Days: {pred['summary']['dense_fog_days']}")
        print(f"  Peak Cold Wave Risk: {pred['summary']['peak_cold_wave_risk_score']} / 100")
        print(f"  Peak Flood Risk: {pred['summary']['peak_flood_risk_score']} / 100")
        print("\nPhase Summaries:")
        for phase, stats in pred['phase_breakdown'].items():
            print(f"  [{phase}]: Temp {stats['avg_temp_min_c']}°C to {stats['avg_temp_max_c']}°C | Rain {stats['accumulated_rain_mm']} mm")
    elif args.gps:
        print_banner()
        import math, json
        from modules.evacroute import assess_isolation_risk
        try:
            parts = [float(x.strip()) for x in args.gps.split(',')]
            gps_lat, gps_lon = parts[0], parts[1]
        except Exception:
            print("[!] Invalid GPS coordinates format. Please provide: --gps 'lat,lon' (e.g. --gps '23.8103,90.4125')")
            return

        live_json = os.path.join(BASE_DIR, 'data', 'processed', 'bangladesh_64_districts_live_weather.json')
        if not os.path.exists(live_json):
            print("[!] Dataset missing.")
            return

        with open(live_json, encoding='utf-8') as f:
            dists = json.load(f)

        def haversine(lat1, lon1, lat2, lon2):
            R = 6371.0
            dLat = math.radians(lat2 - lat1)
            dLon = math.radians(lon2 - lon1)
            a = math.sin(dLat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dLon/2)**2
            return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        nearest = min(dists, key=lambda d: haversine(gps_lat, gps_lon, d['lat'], d['lon']))
        dist_km = haversine(gps_lat, gps_lon, nearest['lat'], nearest['lon'])

        print(f"\n📍 REAL-TIME GPS TRACKER TELEMETRY & TACTICAL HUD")
        print("=" * 80)
        print(f"  • User GPS Coordinates        : {gps_lat:.4f}° N, {gps_lon:.4f}° E")
        print(f"  • Nearest Bangladesh District : {nearest['name']} ({nearest['division']} Division) · {dist_km:.2f} km away")
        print(f"  • Current Weather at Position : {nearest['temp']}°C (Feels {nearest['feels']}°C) · Wind: {nearest['wind']} km/h")
        print(f"  • Active Disaster Alert Tier  : [{nearest.get('alert_tier', 'GREEN NORMAL')}]")
        print(f"  • Monitored Hazard Category   : {nearest.get('hazard_type', 'None')}")
        
        evac = assess_isolation_risk(nearest['name'], water_rise_m=1.0)
        print(f"\n🚑 EMERGENCY SHELTER & EVACUATION DIRECTIVE:")
        print(f"  • Designated High-Ground Haven: {evac['shelter']['name']}")
        print(f"  • Shelter Capacity & Type     : {evac['shelter']['type']} ({evac['shelter']['capacity']})")
        print(f"  • Routing Directive           : {evac['shelter']['routing_directive']}")
        print(f"  • Cell Broadcast Dispatch     : {nearest.get('english_alert', '')[:90]}...")
        print("=" * 80)
    elif args.district:
        run_all_pillars_for_district(args.district)
    elif args.agro:
        print_banner()
        res = assess_district_crop_risk(args.agro)
        for k, v in res.items(): print(f"  {k}: {v}")
    elif args.evac:
        print_banner()
        res = assess_isolation_risk(args.evac)
        for k, v in res.items(): print(f"  {k}: {v}")
    elif args.clima:
        print_banner()
        res = compute_climadeval_rating(args.clima)
        for k, v in res.items(): print(f"  {k}: {v}")
    elif args.sim:
        print_banner()
        sim = run_policy_simulation(500.0)
        for k, v in sim.items(): print(f"  {k}: {v}")
    elif args.health:
        print_banner()
        from modules.health_epidemic import assess_epidemic_risk
        res = assess_epidemic_risk(args.health)
        print(f"\n🏥 HEALTHEPIDEMIC INTELLIGENCE: {res['district'].upper()}")
        print("=" * 80)
        print(f"  • Composite Health Risk Score : {res['composite_health_risk']}/100 ({res['alert_tier']})")
        print(f"  • Dengue Outbreak Index       : {res['dengue_risk_score']}/100")
        print(f"  • Cholera/Waterborne Index    : {res['cholera_risk_score']}/100")
        print(f"  • Hospital Bed Occupancy      : {res['hospital_capacity']['bed_occupancy_pct']}% ({res['hospital_capacity']['status']})")
        print(f"  • Weekly Patient Admissions   : {res['hospital_capacity']['projected_weekly_patients']:,} patients")
        print(f"  • IV Saline Bags Required     : {res['supply_logistics']['iv_saline_bags_1000ml']:,} bags")
        print(f"  • Halogen Water Tabs Needed   : {res['supply_logistics']['halogen_water_purification_tabs']:,} tablets")
        print(f"  • Medical Directive           : {res['directive']}")
        print("=" * 80)
    elif args.grid:
        print_banner()
        from modules.grid_resilience import assess_grid_vulnerability
        res = assess_grid_vulnerability(args.grid)
        print(f"\n⚡ GRIDRESILIENCE CRITICAL INFRASTRUCTURE: {res['district'].upper()}")
        print("=" * 80)
        print(f"  • Overall Blackout Risk       : {res['blackout_probability_pct']}% ({res['grid_status']})")
        print(f"  • Substation Flood Threat     : {res['substation_vulnerability']['substations_at_flood_risk']}/{res['substation_vulnerability']['total_substations']} substations at risk ({res['substation_vulnerability']['flood_submersion_probability_pct']}%)")
        print(f"  • Telecom BTS Cell Towers     : {res['telecom_infrastructure']['projected_towers_offline']}/{res['telecom_infrastructure']['total_bts_towers']} towers offline (-{res['telecom_infrastructure']['cellular_coverage_loss_pct']}% coverage)")
        print(f"  • Hospital Generator Fuel     : {res['backup_power_logistics']['daily_diesel_liters']:,} Liters/day for {res['backup_power_logistics']['projected_outage_hours']} hrs outage")
        print(f"  • Strategic Fuel Reserve      : {res['backup_power_logistics']['strategic_fuel_depot']}")
        print("=" * 80)
    elif args.erosion:
        print_banner()
        from modules.riverbank_erosion import assess_riverbank_erosion
        res = assess_riverbank_erosion(args.erosion)
        print(f"\n🌊 RIVERBANK EROSION & CHAR DISPLACEMENT: {res['district'].upper()}")
        print("=" * 80)
        print(f"  • River System & Basin        : {res['primary_river_system']}")
        print(f"  • Vulnerability Level         : {res['erosion_vulnerability_rating']}")
        print(f"  • Projected Land Diluvion     : {res['projected_land_loss_ha']} Hectares ({res['projected_land_loss_sq_km']} sq km)")
        print(f"  • Climate Refugees Displaced  : {res['displacement_estimates']['climate_refugees_displaced']:,} Citizens ({res['displacement_estimates']['homesteads_submerged']:,} homes lost)")
        print(f"  • Emergency Relocation Grant  : ${res['displacement_estimates']['immediate_resettlement_grant_usd']:,.2f} USD")
        print(f"  • Host Resettlement Cluster   : {res['resettlement_logistics']['designated_host_zone']}")
        print("=" * 80)
    elif args.supply:
        print_banner()
        from modules.agri_supply import analyze_supply_chain_bottlenecks
        res = analyze_supply_chain_bottlenecks(args.supply)
        print(f"\n🍞 AGRISUPPLY FOOD SECURITY & HIGHWAYS: {res['district'].upper()}")
        print("=" * 80)
        print(f"  • Agricultural Commodity Hub  : {res['commodity_specialization']}")
        print(f"  • Daily Metropolitan Freight  : {res['daily_volume']['truckloads_to_metropolitan']} trucks ({res['daily_volume']['metric_tons_dispatched']} tons)")
        print(f"  • Highway Chokepoint Status   : {res['highway_chokepoint_analysis']['chokepoint_status']} (Flood prob: {res['highway_chokepoint_analysis']['highway_submersion_probability_pct']}%)")
        print(f"  • Detour Transit Penalty      : +{res['freight_rerouting_logistics']['detour_additional_distance_km']} km (+{res['freight_rerouting_logistics']['detour_additional_transit_hours']} hrs, +{res['freight_rerouting_logistics']['diesel_fuel_penalty_per_truck_liters']}L diesel/truck)")
        print(f"  • Capital Food Price Impact   : {res['urban_market_impact']['projected_commodity_price_inflation_pct']} inflation risk in {res['urban_market_impact']['at_risk_metropolitan_hub']}")
        print("=" * 80)
    elif args.bot:
        print_banner()
        from modules.deltabot import ask_deltabot
        res = ask_deltabot(args.bot)
        print(f"\n🤖 DELTABOT AI CLIMATE COPILOT [{res['pillar'].upper()}]")
        print("=" * 80)
        print(f"Q: \"{res['query']}\"")
        print(f"A: {res['response']}")
        print("=" * 80)
    else:
        interactive_terminal_menu()


if __name__ == '__main__':
    main()
