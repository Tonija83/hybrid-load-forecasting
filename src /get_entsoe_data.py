from entsoe import EntsoePandasClient
import pandas as pd
from datetime import datetime, timedelta

# API KEY aus deinem ENTSO-E Konto (nach Registrierung)
API_KEY = 'DEIN_API_KEY_HIER'

client = EntsoePandasClient(api_key=API_KEY)

# Beispiel: Daten für Österreich (AT)
country_code = 'AT'

# Zeitbereich definieren
start = pd.Timestamp('2024-07-01', tz='Europe/Vienna')
end = pd.Timestamp('2024-07-07', tz='Europe/Vienna')

# Verbrauchsdaten (Load)
load = client.query_load(country_code, start=start, end=end)

# Erzeugung nach Technologie
gen_mix = client.query_generation_mix(country_code, start=start, end=end)

# Speicherung als CSV
load.to_csv('data/entsoe_load_at.csv')
gen_mix.to_csv('data/entsoe_generation_mix_at.csv')

print("Daten erfolgreich gespeichert.")
