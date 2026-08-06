from pathlib import Path
import random
import secrets
import string
from tqdm import tqdm
from korean_name_generator import namer
from datetime import datetime, timedelta

from src.utils.io import read_md, read_xlsx
from src.config.ontology import *



def _get_timestamp():
    start = datetime(2023, 1, 1, 0, 0, 0)
    end = datetime(2026, 12, 31, 23, 59, 59)

    random_seconds = random.randint(0, int((end - start).total_seconds()))
    dt = start + timedelta(seconds=random_seconds)

    return dt.strftime("%Y.%m.%d %H:%M:%S")


def _get_rand_str(k):
    chars = string.ascii_letters + string.digits
    return ''.join(secrets.choice(chars) for _ in range(k))


def _get_name():
    gender = random.choice(gender_pool)
    name = namer.generate(gender)

    return name


def _get_blind_name(name):
    return f"{name[0]}*{name[-1]}"


def _get_address():
    delivery_location_pool = list(address_pool.keys())
    delivery_location = random.choice(delivery_location_pool)
    address = random.choice(address_pool[delivery_location])

    road = random.choice(["길 ", "번길 ", "-", " - "])
    x1 = random.randrange(1,42)
    x2 = random.randrange(1,42)

    full_address = f"{delivery_location[:3]} {delivery_location[2:]}구 {address} {x1}{road}{x2}".replace("구구", "구")
    return delivery_location, full_address


def build_dummy_data(md_dir_path: str, k_for_md:int = 1000) -> list[str]:
    df_products = read_xlsx("./data/raw/서울상품_20260213.xlsx")
    products = {}

    for _, row in df_products.iterrows():
        if row["업체명"] in products:
            products[row["업체명"]].append(row["이름"])
        else:
            products[row["업체명"]] = [row["이름"]]

    df_police_precinct = read_xlsx("./data/raw/260806_전국경찰관서목록.xlsx")
    police_precinct_pool = df_police_precinct["관서"].tolist()

    mds = list(Path(md_dir_path).glob("*.md"))
    results = []

    for md_path in tqdm(mds):
        for _ in range(k_for_md):
            user = _get_name()
            driver_name = _get_name()
            blind_user = _get_blind_name(user)

            delivery_location, address = _get_address()

            shipment_number = "".join(random.choices("0123456789", k=14))
            rand4_1 = "".join(random.choices("0123456789", k=4))
            rand4_2 = "".join(random.choices("0123456789", k=4))
            rand6_1 = "".join(random.choices("0123456789", k=6))
            rand1_1 = random.randrange(1,9)
            rand11_str = _get_rand_str(11)
            rand12_str = _get_rand_str(12)

            lotte_empno = "".join(random.choices("0123456789", k=8))
            cj_empnum = _get_rand_str(8)
            cj_trspbillnum = _get_rand_str(16)

            prosecution = random.choice(prosecution_pool)
            police_precinct = random.choice(police_precinct_pool)
            crime_name = random.choice(crime_name_pool)

            retail_shop = random.choice(list(products.keys()))
            product_name = random.choice(products[retail_shop])
            delivery_company = random.choice(delivery_company_pool)
            cp_order_number = "".join(random.choices("0123456789", k=14))

            year = random.choice(year_pool)
            time_month2 = random.choice(time_month2_pool)
            time_day2 = random.choice(time_day2_pool)
            time_hour = random.choice(time_hour_pool)
            timestamp = _get_timestamp()


            inputs = {
                "user": user,
                "blind_user": blind_user,
                "driver_name": driver_name,

                "delivery_location": delivery_location,
                "address": address,

                "shipment_number": shipment_number,
                "rand4_1": rand4_1,
                "rand4_2": rand4_2,
                "rand6_1": rand6_1,
                "rand1_1": rand1_1,
                "rand11_str": rand11_str,
                "rand12_str": rand12_str,

                "lotte_empno": lotte_empno,
                "cj_empnum": cj_empnum,
                "cj_trspbillnum": cj_trspbillnum,

                "prosecution": prosecution,
                "police_precinct": police_precinct,
                "crime_name": crime_name,
                "police_call_number": random.choice([f"0{rand1_1}{rand4_2}{rand4_1}", f"02-{rand4_2}-{rand4_1}", f'0{"".join(random.choices("3456", k=1))}{"".join(random.choices("12345", k=1))}-{rand4_2[1:]}-{rand4_1}']),

                "retail_shop": retail_shop,
                "product_name": product_name,
                "delivery_company": delivery_company,
                "cp_order_number": cp_order_number,

                "year": year,
                "time_month2": time_month2,
                "time_day2": time_day2,
                "time_hour": time_hour,
                "timestamp": timestamp,
            }    


            '''
            내부 dict 형태
            {
                "label": "0",
                "full_text": str,
                "urls": list[str],
                "detail: str # 출처
            }
            '''
            full_text = read_md(str(md_path), **inputs)
            results.append(full_text)

    return results