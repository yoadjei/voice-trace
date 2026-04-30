# generates 120 epidemiologically grounded english injury narratives via claude.
# saves to data/synthetic/narratives_en.csv
# run: python -m data_gen.generate_narratives
import json
import os
import time
import random
import pandas as pd
import anthropic
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DISTRIBUTIONS_PATH = Path("data/distributions/ghana_injury_distributions.json")
OUTPUT_PATH = Path("data/synthetic/narratives_en.csv")

RATE_LIMIT_DELAY = 13  # seconds — free tier: 5 req/min

# ghanaian names by sex — akan, ga, ewe, dagbani, hausa, nzema, fante, gonja
NAMES_MALE = [
    "Kofi", "Kwame", "Kweku", "Kwabena", "Yaw", "Kojo", "Kwasi",
    "Nii", "Nii Armah", "Nii Kwei", "Nii Laryea",
    "Dela", "Selorm", "Mawuli", "Edem", "Kpodo",
    "Fuseini", "Alhassan", "Sulemana", "Ibrahim", "Yakubu", "Issah",
    "Abubakar", "Hamza", "Muntari",
    "Adjei", "Asante", "Boateng", "Mensah", "Appiah", "Ofori",
    "Agyeman", "Darkwa", "Asare", "Owusu",
    "Baffour", "Amankwah", "Wiredu", "Amoah",
]

NAMES_FEMALE = [
    "Ama", "Akosua", "Abena", "Adwoa", "Akua", "Afia", "Araba",
    "Naa", "Naa Akorkor", "Naa Dedei", "Naa Shormeh",
    "Esi", "Efua", "Ekua", "Ewurama",
    "Vida", "Gifty", "Patience", "Comfort", "Abigail", "Felicia",
    "Fatima", "Mariama", "Ramatu", "Memunatu", "Hadija",
    "Afia", "Adjoa", "Maame", "Yaa", "Akosua",
    "Emefa", "Sena", "Dzifa", "Elinam",
    "Abenaa", "Nhyira", "Serwa",
]

CAMPUS_LOCATIONS = [
    # universities
    "KNUST campus, Kumasi", "University of Ghana, Legon", "UDS main campus, Tamale",
    "Cape Coast University campus", "UENR campus, Sunyani", "UMaT, Tarkwa",
    "Ghana Communication Technology University, Accra", "Ashesi University, Berekuso",
    "UPSA campus, Accra", "Central University, Miotso", "Valley View University, Oyibi",
    "Ho Technical University", "Koforidua Technical University",
    "Accra Technical University", "Tamale Technical University",
    # senior high schools
    "Achimota Senior High School, Accra", "Prempeh College, Kumasi",
    "Mfantsipim School, Cape Coast", "Wesley Girls, Cape Coast",
    "Opoku Ware School, Kumasi", "GSTS, Takoradi",
    "Adisadel College, Cape Coast", "Presec Legon, Accra",
    "T.I. Amass, Kumasi", "Kumasi Academy",
    "Tamale Senior High School", "Bolgatanga Senior High School",
    "Ho Senior High School", "Koforidua Senior High Technical",
    "Winneba Senior High School", "Navrongo Senior High School",
    "Yendi Senior High School", "Wa Senior High School",
]

CAMPUS_PROMPT = """Generate a naturalistic spoken injury report (2-5 sentences) as if a student, teacher, or parent is calling a health hotline to report an incident that happened on a school or university campus in Ghana.

Details to include:
- Caller's name: {caller_name}
- Victim's name: {victim_name}
- Injury type: {injury_type}
- Victim: {age_group}, {sex}
- Severity: {severity}
- Body region affected: {body_region}
- Location: {location}

Guidelines:
- This is a campus incident — be creative and varied. Examples: fight between students, stabbing after argument, relationship dispute that turned violent (including between girls), sports injury during training or match, hazing/initiation gone wrong, fall from bunk bed or staircase, lab chemical accident, fainting from stress or heat, canteen accident, bullying that got physical
- Caller can be a fellow student, roommate, teacher, house prefect, hostel master, canteen worker, security guard, or parent who got a call
- Write in simple informal English — not medically trained
- Use the names naturally (not in every sentence)
- Vary how the caller opens the report
- Vary sentence rhythm — not all sentences the same length
- Do not use medical jargon, no greeting to the listener, no sign-off"""

# locations across all 16 regions of ghana
LOCATION_EXAMPLES = {
    "highway": [
        # greater accra / eastern
        "Accra-Kumasi highway near Nsawam", "Tema Motorway near Ashaiman",
        "N1 highway near Kasoa", "Accra-Cape Coast road near Winneba",
        "Accra-Aflao road near Sogakope",
        # ashanti / bono
        "Kumasi-Sunyani road near Techiman", "Kumasi-Tamale road near Ejura",
        "Kumasi-Cape Coast road near Mankessim",
        # northern / savannah / north east
        "Tamale-Bolgatanga road near Walewale", "Accra-Kumasi-Tamale corridor near Kintampo",
        "Tamale-Yendi road near Savelugu", "Bawku road near Zebilla",
        # volta / oti
        "Ho-Accra road near Adidome", "Hohoe-Jasikan road", "Kete Krachi road near Dambai",
        # western / central
        "Takoradi-Cape Coast highway near Anomabo", "Axim road near Half Assini",
        "Cape Coast-Elmina road",
        # upper east / upper west
        "Bolgatanga-Bawku road near Navrongo", "Wa-Kumasi road near Sawla",
        "Tumu-Wa road near Funsi",
    ],
    "urban_road": [
        # greater accra
        "Nima, Accra", "Madina, Accra", "Labadi, Accra", "Teshie, Accra",
        "Korle-Bu, Accra", "Kaneshie, Accra", "Lapaz, Accra", "Dansoman, Accra",
        "Spintex Road, Accra", "Adenta, Accra", "Tema Community 1",
        # ashanti
        "Adum, Kumasi", "Suame, Kumasi", "Asokwa, Kumasi", "Bantama, Kumasi",
        "Nhyiaeso, Kumasi", "Oforikrom, Kumasi",
        # western
        "Takoradi market area", "Sekondi", "Axim town", "Tarkwa central",
        # central
        "Cape Coast town", "Elmina", "Saltpond", "Assin Fosu",
        # eastern
        "Koforidua town centre", "Nkawkaw", "Suhum", "Kade",
        # volta / oti
        "Ho town centre", "Hohoe", "Kpando", "Dambai", "Jasikan",
        # northern / savannah
        "Tamale central", "Sagnarigu, Tamale", "Yendi", "Damongo", "Salaga",
        # upper east
        "Bolgatanga", "Bawku", "Navrongo", "Zebilla",
        # upper west
        "Wa central", "Tumu", "Lawra",
        # bono / ahafo
        "Sunyani", "Techiman", "Dormaa Ahenkro", "Goaso", "Kenyasi",
        # north east
        "Nalerigu", "Gambaga",
    ],
    "home": [
        # greater accra
        "their home in Tesano", "a compound house in Agbogbloshie",
        "a house in Bubuashie", "their home in Mamprobi", "a flat in Adenta",
        "their room in Darkuman", "a house in Ashiaman",
        # ashanti
        "a house in Suame, Kumasi", "their home in Asokwa", "a compound in Dichemso",
        "their home in Bekwai", "a house in Obuasi",
        # northern / upper east / upper west
        "their compound in Bolgatanga", "a family house in Navrongo",
        "their home in Wa", "a compound in Tumu", "their house in Bawku",
        "a family compound in Tamale", "their home in Yendi",
        # volta
        "their home in Ho", "a house in Hohoe", "their compound in Kpando",
        # western
        "their home in Takoradi", "a house in Axim", "their home in Tarkwa",
        # central
        "their home in Cape Coast", "a house in Elmina",
        # eastern
        "their home in Koforidua", "a house in Nkawkaw",
        # bono
        "their home in Sunyani", "a house in Techiman",
    ],
}


GENERATION_PROMPT = """Generate a naturalistic spoken injury report (2-5 sentences) as if a community member in Ghana is calling a health hotline to report what happened.

Details to include:
- Caller's name: {caller_name} (caller is reporting about someone else, or themselves)
- Victim's name: {victim_name}
- Injury type: {injury_type}
- Victim: {age_group}, {sex}
- Severity: {severity}
- Body region affected: {body_region}
- Location: {location}

Guidelines:
- Write in simple informal English, as if the caller is not medically trained
- The caller can be the victim, a family member, bystander, or neighbor — vary this naturally
- Use the names naturally in the report (not every sentence)
- Vary how the caller opens (e.g. "Please...", "Good afternoon, I want to report...", "Something happened...", "I'm calling because...", "My [relation]...")
- Vary sentence rhythm and length — not all sentences the same length
- Do not use medical jargon
- Do not include a greeting addressed to the listener or a sign-off — just the spoken report
- For road traffic incidents (RTA), vary the vehicle type naturally — do NOT default to motorcycle. Use the full range of Ghanaian road transport: trotro, bus, shared taxi (dropping), OA (Opel Astra taxi), private car, pickup truck, tipper truck, articulated truck, pragyia (tricycle/Kia), aboboyaa (motorised cargo tricycle), okada (motorbike taxi), bicycle, canoe on a river crossing — choose whichever fits the location and context"""


def _sample_row(distributions: dict, rng: random.Random, campus: bool = False) -> dict:
    # weighted random sample of one narrative's metadata
    def weighted_choice(d):
        keys, weights = zip(*d.items())
        return rng.choices(keys, weights=weights, k=1)[0]

    sex = weighted_choice(distributions["sex"])
    name_pool = NAMES_MALE if sex == "male" else NAMES_FEMALE
    victim_name = rng.choice(name_pool)
    caller_name = rng.choice(NAMES_MALE + NAMES_FEMALE)

    injury_type = weighted_choice(distributions["injury_type"])
    age_group = weighted_choice(distributions["age_group"])
    severity = weighted_choice(distributions["severity"])
    body_region = weighted_choice(distributions["body_region"])

    if campus:
        location_type = "campus"
        location = rng.choice(CAMPUS_LOCATIONS)
    else:
        location_type = weighted_choice(distributions["location_type"])
        location = rng.choice(LOCATION_EXAMPLES[location_type])

    return {
        "injury_type": injury_type,
        "sex": sex,
        "age_group": age_group,
        "severity": severity,
        "body_region": body_region,
        "location_type": location_type,
        "location": location,
        "victim_name": victim_name,
        "caller_name": caller_name,
    }


def generate_narratives(n: int = 120, seed: int = 42) -> pd.DataFrame:
    # generate n narratives, resumable — skips rows already saved to disk
    rng = random.Random(seed)
    distributions = json.loads(DISTRIBUTIONS_PATH.read_text())
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # load existing rows so we can resume mid-run
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        # only skip rows that already have a narrative — retry empty ones
        done_ids = set(existing.loc[existing["narrative_en"].notna() & (existing["narrative_en"] != ""), "id"].tolist())
        rows = {r["id"]: r for r in existing.to_dict("records")}
        print(f"resuming: {len(done_ids)} done, {n - len(done_ids)} remaining")
    else:
        done_ids = set()
        rows = {}

    for i in range(n):
        is_campus = i >= 100
        if i in done_ids:
            # advance rng so seed stays consistent
            _sample_row(distributions, rng, campus=is_campus)
            continue

        meta = _sample_row(distributions, rng, campus=is_campus)
        active_prompt = CAMPUS_PROMPT if is_campus else GENERATION_PROMPT
        prompt = active_prompt.format(
            caller_name=meta["caller_name"],
            victim_name=meta["victim_name"],
            injury_type=meta["injury_type"].upper(),
            sex=meta["sex"],
            age_group=meta["age_group"],
            severity=meta["severity"],
            body_region=meta["body_region"].replace("_", "/"),
            location=meta["location"],
        )
        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            narrative = response.content[0].text.strip()
        except Exception as e:
            print(f"[generate] ERROR on row {i}: {e}")
            narrative = ""

        row = {"id": i, "narrative_en": narrative, **meta}
        rows[i] = row

        # save after each row so progress is never lost
        pd.DataFrame(list(rows.values())).sort_values("id").to_csv(OUTPUT_PATH, index=False)
        status = "ok" if narrative else "EMPTY"
        done_count = sum(1 for r in rows.values() if r.get("narrative_en"))
        print(f"[{i+1}/{n}] [{status}] {meta['injury_type']} | {meta['sex']} | {meta['age_group']} | {meta['location']} — {done_count}/{n} complete, sleeping {RATE_LIMIT_DELAY}s")
        time.sleep(RATE_LIMIT_DELAY)

    df = pd.DataFrame(list(rows.values())).sort_values("id").reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)
    empty = (df["narrative_en"] == "").sum() + df["narrative_en"].isna().sum()
    print(f"\ndone. {len(df) - empty}/{n} narratives generated, {empty} empty — saved to {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    generate_narratives()
