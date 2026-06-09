"""Agent1 식재료 표준명 매핑.

Detector와 LLM이 반환할 수 있는 영어/한국어 표현을 다음 에이전트가 쓰는
한국어 식재료 표준명으로 정리한다.
"""

from __future__ import annotations


STANDARD_INGREDIENT_GROUPS: dict[str, tuple[str, ...]] = {
    # 탄수화물/곡류
    "밥": ("cooked rice", "steamed rice", "leftover rice", "white rice", "rice bowl", "쌀밥", "흰쌀밥", "찬밥"),
    "쌀": ("rice", "uncooked rice", "백미"),
    "현미": ("brown rice",),
    "보리": ("barley",),
    "밀가루": ("flour", "wheat flour"),
    "전분": ("starch", "potato starch", "corn starch", "옥수수전분", "감자전분"),
    "빵": ("bread", "toast", "sandwich bread", "식빵", "토스트"),
    "면": ("noodle", "noodles", "국수", "소면"),
    "라면": ("ramen", "instant noodle", "instant noodles"),
    "우동면": ("udon", "udon noodle", "udon noodles"),
    "파스타면": ("pasta", "spaghetti", "spaghetti noodles"),
    "피자도우": ("pizza dough", "dough", "pizza crust"),
    "떡": ("rice cake", "tteok", "떡국떡", "떡볶이떡"),
    "감자": ("potato", "potatoes"),
    "고구마": ("sweet potato", "sweet potatoes"),
    "옥수수": ("corn", "sweet corn"),
    # 단백질/육류
    "계란": ("egg", "eggs", "달걀", "계란후라이", "fried egg", "boiled egg"),
    "소고기": ("beef", "소 고기", "쇠고기", "ground beef", "minced beef"),
    "돼지고기": ("pork", "돼지 고기", "ground pork", "minced pork", "pork belly", "삼겹살", "목살"),
    "닭고기": ("chicken", "닭 고기", "chicken breast", "닭가슴살", "닭다리"),
    "오리고기": ("duck", "duck meat"),
    "양고기": ("lamb", "mutton"),
    "햄": ("ham",),
    "베이컨": ("bacon",),
    "소시지": ("sausage", "sausages", "소세지"),
    "페퍼로니": ("pepperoni", "pepperoni slices"),
    "참치": ("tuna", "canned tuna", "참치캔"),
    "연어": ("salmon",),
    "고등어": ("mackerel",),
    "갈치": ("hairtail", "beltfish"),
    "명태": ("pollock",),
    "대구": ("cod",),
    "멸치": ("anchovy", "anchovies"),
    "새우": ("shrimp", "prawn", "prawns"),
    "오징어": ("squid",),
    "문어": ("octopus",),
    "낙지": ("small octopus", "nakji"),
    "게": ("crab",),
    "바지락": ("clam", "clams", "manila clam"),
    "홍합": ("mussel", "mussels"),
    "굴": ("oyster", "oysters"),
    "어묵": ("fish cake", "fishcake", "eomuk"),
    "두부": ("tofu", "bean curd"),
    "순두부": ("soft tofu", "silken tofu"),
    "콩": ("bean", "beans", "soybean", "soybeans"),
    # 채소
    "양파": ("onion", "onions"),
    "대파": ("green onion", "green onions", "scallion", "scallions", "green leek", "leek", "파"),
    "쪽파": ("chive", "chives", "small green onion"),
    "부추": ("garlic chive", "garlic chives", "buchu"),
    "마늘": ("garlic", "garlic clove", "garlic cloves"),
    "다진마늘": ("minced garlic", "crushed garlic", "간마늘"),
    "생강": ("ginger",),
    "당근": ("carrot", "carrots"),
    "양배추": ("cabbage", "green cabbage"),
    "배추": ("napa cabbage", "chinese cabbage"),
    "청경채": ("bok choy", "pak choi", "bokchoi"),
    "상추": ("lettuce",),
    "깻잎": ("perilla leaf", "perilla leaves"),
    "시금치": ("spinach",),
    "바질": ("basil", "basil leaf", "basil leaves", "fresh basil"),
    "콩나물": ("bean sprout", "bean sprouts", "soybean sprout", "soybean sprouts"),
    "숙주": ("mung bean sprout", "mung bean sprouts"),
    "오이": ("cucumber", "cucumbers"),
    "애호박": ("zucchini", "courgette", "green squash"),
    "호박": ("pumpkin", "squash"),
    "가지": ("eggplant", "aubergine"),
    "토마토": ("tomato", "tomatoes"),
    "방울토마토": ("cherry tomato", "cherry tomatoes"),
    "고추": ("chili pepper", "chilli pepper", "green chili", "red chili", "pepper"),
    "청양고추": ("cheongyang chili", "cheongyang pepper"),
    "피망": ("bell pepper", "green bell pepper"),
    "파프리카": ("paprika", "sweet pepper", "red bell pepper", "yellow bell pepper"),
    "브로콜리": ("broccoli",),
    "콜리플라워": ("cauliflower",),
    "샐러리": ("celery",),
    "무": ("radish", "korean radish", "daikon"),
    "연근": ("lotus root",),
    "우엉": ("burdock root",),
    "아스파라거스": ("asparagus",),
    "아보카도": ("avocado",),
    "김치": ("kimchi",),
    "파김치": ("green onion kimchi",),
    "채소": ("vegetable", "vegetables", "green vegetable", "unclear green vegetable"),
    # 버섯/해조류
    "버섯": ("mushroom", "mushrooms"),
    "느타리버섯": ("oyster mushroom", "oyster mushrooms", "굴버섯"),
    "새송이버섯": ("king oyster mushroom", "king oyster mushrooms"),
    "팽이버섯": ("enoki mushroom", "enoki mushrooms"),
    "표고버섯": ("shiitake mushroom", "shiitake mushrooms", "shiitake"),
    "양송이버섯": ("button mushroom", "button mushrooms", "white mushroom", "white mushrooms"),
    "목이버섯": ("wood ear mushroom", "wood ear mushrooms", "black fungus"),
    "김": ("seaweed", "gim", "nori"),
    "미역": ("wakame", "sea mustard"),
    "다시마": ("kelp", "kombu"),
    # 과일
    "사과": ("apple", "apples"),
    "배": ("pear", "asian pear"),
    "바나나": ("banana", "bananas"),
    "딸기": ("strawberry", "strawberries"),
    "블루베리": ("blueberry", "blueberries"),
    "레몬": ("lemon", "lemons"),
    "라임": ("lime", "limes"),
    "오렌지": ("orange", "oranges"),
    # 유제품/지방
    "우유": ("milk",),
    "버터": ("butter",),
    "치즈": ("cheese",),
    "모짜렐라치즈": (
        "mozzarella",
        "mozzarella cheese",
        "shredded mozzarella",
        "shredded mozzarella cheese",
        "shredded cheese",
    ),
    "생크림": ("cream", "heavy cream", "whipping cream"),
    "요거트": ("yogurt", "yoghurt"),
    "식용유": ("oil", "cooking oil", "vegetable oil"),
    "올리브유": ("olive oil",),
    "참기름": ("sesame oil",),
    "들기름": ("perilla oil",),
    # 양념/소스
    "소금": ("salt",),
    "설탕": ("sugar",),
    "후추": ("black pepper", "pepper powder"),
    "간장": ("soy sauce",),
    "국간장": ("soup soy sauce",),
    "진간장": ("dark soy sauce",),
    "된장": ("doenjang", "soybean paste"),
    "고추장": ("gochujang", "red pepper paste"),
    "쌈장": ("ssamjang",),
    "고춧가루": ("gochugaru", "red pepper powder", "chili flakes", "chilli flakes"),
    "식초": ("vinegar",),
    "맛술": ("mirin", "cooking wine"),
    "청주": ("rice wine", "sake"),
    "굴소스": ("oyster sauce",),
    "액젓": ("fish sauce",),
    "케첩": ("ketchup", "catsup"),
    "토마토소스": ("tomato sauce", "pizza sauce", "marinara sauce", "red sauce"),
    "마요네즈": ("mayonnaise", "mayo"),
    "머스타드": ("mustard",),
    "카레가루": ("curry powder",),
    "꿀": ("honey",),
    "올리고당": ("oligosaccharide syrup", "corn syrup", "물엿"),
    "깨": ("sesame seed", "sesame seeds"),
    "땅콩": ("peanut", "peanuts"),
    "아몬드": ("almond", "almonds"),
    "호두": ("walnut", "walnuts"),
    "올리브": ("olive", "olives", "black olive", "black olives"),
    # 식재료가 아니거나 너무 넓은 detector 라벨
    "병": ("bottle", "jar"),
    "음식": ("food",),
    "음료": ("drink", "beverage"),
}

STANDARD_INGREDIENT_ALIASES: dict[str, str] = {}
for standard_name, aliases in STANDARD_INGREDIENT_GROUPS.items():
    STANDARD_INGREDIENT_ALIASES[standard_name.lower()] = standard_name
    for alias in aliases:
        STANDARD_INGREDIENT_ALIASES[alias.lower()] = standard_name

GENERIC_DETECTION_LABELS = {"food", "vegetable", "vegetables", "bottle", "jar", "drink", "beverage", "음식", "채소", "병", "음료"}

CONFIRMATION_CANDIDATES: dict[str, tuple[str, ...]] = {
    "vegetable": ("청경채", "대파", "양파", "고추", "양배추", "애호박"),
    "vegetables": ("청경채", "대파", "양파", "고추", "양배추", "애호박"),
    "green vegetable": ("바질", "청경채", "시금치", "상추", "깻잎", "대파"),
    "채소": ("청경채", "대파", "양파", "고추", "양배추", "애호박"),
    "mushroom": ("느타리버섯", "팽이버섯", "새송이버섯", "표고버섯", "양송이버섯"),
    "버섯": ("느타리버섯", "팽이버섯", "새송이버섯", "표고버섯", "양송이버섯"),
    "green onion": ("대파", "쪽파", "부추"),
    "green leek": ("대파", "쪽파", "부추"),
    "scallion": ("대파", "쪽파", "부추"),
    "대파": ("대파", "쪽파", "부추"),
    "chili pepper": ("고추", "청양고추", "피망", "파프리카"),
    "고추": ("고추", "청양고추", "피망", "파프리카"),
    "bottle": ("간장", "식초", "참기름", "식용유", "맛술"),
    "jar": ("고추장", "된장", "쌈장", "잼"),
    "tomato sauce": ("토마토소스", "케첩", "고추장"),
    "병": ("간장", "식초", "참기름", "식용유", "맛술"),
}

DEFAULT_DETECTOR_LABELS = [
    "green onion",
    "scallion",
    "green leek",
    "tomato",
    "beef",
    "pork",
    "chicken",
    "tofu",
    "mushroom",
    "enoki mushroom",
    "oyster mushroom",
    "king oyster mushroom",
    "shiitake mushroom",
    "cabbage",
    "napa cabbage",
    "bean sprouts",
    "bok choy",
    "zucchini",
    "egg",
    "onion",
    "garlic",
    "potato",
    "carrot",
    "radish",
    "spinach",
    "basil",
    "fresh basil",
    "lettuce",
    "cucumber",
    "chili pepper",
    "bell pepper",
    "green bell pepper",
    "red bell pepper",
    "pizza dough",
    "dough",
    "pizza sauce",
    "tomato sauce",
    "mozzarella cheese",
    "shredded mozzarella cheese",
    "shredded cheese",
    "pepperoni",
    "black olives",
    "soy sauce",
    "gochujang",
    "doenjang",
    "sesame oil",
    "bottle",
    "seaweed",
    "kelp",
    "vegetable",
]


def standardize_ingredient_name(name: str) -> str:
    cleaned = " ".join(name.strip().split())
    return STANDARD_INGREDIENT_ALIASES.get(cleaned.lower(), cleaned)


def build_standard_name_prompt_rules() -> str:
    lines = []
    for standard_name, aliases in STANDARD_INGREDIENT_GROUPS.items():
        alias_text = ", ".join(aliases[:6])
        if alias_text:
            lines.append(f"- {alias_text} => {standard_name}")
    return "\n".join(lines)


def suggest_confirmation_candidates(name: str, original_label: str = "") -> list[str]:
    keys = [
        original_label.strip().lower(),
        name.strip().lower(),
        standardize_ingredient_name(name).lower(),
    ]
    candidates: list[str] = []

    for key in keys:
        for candidate in CONFIRMATION_CANDIDATES.get(key, ()):
            if candidate not in candidates:
                candidates.append(candidate)

    standard_name = standardize_ingredient_name(name)
    if standard_name and standard_name not in candidates:
        candidates.insert(0, standard_name)

    return candidates[:6]
