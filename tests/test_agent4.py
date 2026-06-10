import unittest

from agents.agent4.service import route_recipe, route_recipe_node


def egg_rice_candidate():
    return {
        "name": "egg_rice",
        "recipe_type": "korean",
        "core_ingredients": ["rice", "egg"],
        "required_ingredients": ["rice", "egg"],
        "optional_ingredients": ["green_onion"],
        "seasonings": ["soy_sauce"],
        "substitutions": {"green_onion": None},
        "difficulty": "easy",
        "cooking_time_minutes": 10,
        "cooking_methods": ["pan"],
    }


def korean_egg_fried_rice_candidate():
    return {
        "name": "계란볶음밥",
        "recipe_type": "korean",
        "core_ingredients": ["밥", "계란"],
        "required_ingredients": ["밥", "계란"],
        "optional_ingredients": ["대파"],
        "seasonings": ["간장"],
        "substitutions": {"대파": None},
        "difficulty": "easy",
        "cooking_time_minutes": 10,
        "cooking_methods": ["팬 조리"],
    }


class Agent4RecipeRouterTest(unittest.TestCase):
    def test_can_cook_with_complete_candidate_ingredients(self):
        result = route_recipe(
            {
                "available_ingredients": ["rice", "egg", "green_onion", "soy_sauce"],
                "recipe_type": "korean",
                "candidate_foods": [egg_rice_candidate()],
                "food_directions": {
                    "difficulty": "easy",
                    "cooking_time_limit_minutes": 15,
                    "preferred_cooking_method": "pan",
                },
            }
        )

        self.assertEqual(result.route, "can_cook")
        self.assertTrue(result.can_pass_to_agent5)
        self.assertEqual(result.selected_recipe.name, "egg_rice")
        self.assertEqual(result.ingredients_to_use, ["rice", "egg", "green_onion"])
        self.assertEqual(result.seasonings_to_use, ["soy_sauce"])

    def test_simple_when_optional_ingredient_can_be_omitted(self):
        result = route_recipe(
            {
                "available_ingredients": ["rice", "egg", "soy_sauce"],
                "recipe_type": "korean",
                "candidate_foods": [egg_rice_candidate()],
                "food_directions": {
                    "difficulty": "easy",
                    "cooking_time_limit_minutes": 15,
                },
            }
        )

        self.assertEqual(result.route, "simple")
        self.assertTrue(result.can_pass_to_agent5)
        self.assertEqual(result.ingredients_to_use, ["rice", "egg"])
        self.assertEqual(result.substitutions[0].original, "green_onion")

    def test_no_ingredient_passes_only_when_additional_ingredients_allowed(self):
        result = route_recipe(
            {
                "available_ingredients": ["rice", "soy_sauce"],
                "recipe_type": "korean",
                "candidate_foods": [egg_rice_candidate()],
                "ingredient_policy": "allow_additional",
                "food_directions": {
                    "difficulty": "easy",
                    "cooking_time_limit_minutes": 15,
                },
            }
        )

        self.assertEqual(result.route, "no_ingredient")
        self.assertTrue(result.can_pass_to_agent5)
        self.assertEqual(result.additional_ingredients, ["egg"])
        self.assertEqual(result.selected_recipe.name, "egg_rice")

    def test_no_ingredient_blocks_when_only_available_is_selected(self):
        result = route_recipe(
            {
                "available_ingredients": ["rice", "soy_sauce"],
                "recipe_type": "korean",
                "candidate_foods": [egg_rice_candidate()],
                "ingredient_policy": "only_available",
                "food_directions": {
                    "difficulty": "easy",
                    "cooking_time_limit_minutes": 15,
                },
            }
        )

        self.assertEqual(result.route, "no_ingredient")
        self.assertFalse(result.can_pass_to_agent5)
        self.assertIsNone(result.selected_recipe)

    def test_soft_conflict_still_passes_when_core_ingredients_exist(self):
        result = route_recipe(
            {
                "available_ingredients": ["rice", "egg", "soy_sauce"],
                "recipe_type": "korean",
                "candidate_foods": [egg_rice_candidate()],
                "food_directions": {
                    "difficulty": "easy",
                    "fatigue_level": "high",
                    "cooking_time_limit_minutes": 5,
                    "preferred_cooking_method": "oven",
                },
            }
        )

        self.assertEqual(result.route, "simple")
        self.assertTrue(result.can_pass_to_agent5)
        self.assertEqual(result.selected_recipe.name, "egg_rice")

    def test_quick_cooking_preference_matches_simple_pan_candidates(self):
        result = route_recipe(
            {
                "available_ingredients": ["밥", "계란", "간장"],
                "recipe_type": "korean",
                "candidate_foods": [korean_egg_fried_rice_candidate()],
                "food_directions": {
                    "difficulty": "easy",
                    "fatigue_level": "high",
                    "preferred_cooking_method": "빠른 조리",
                    "cooking_time_limit_minutes": 15,
                },
            }
        )

        self.assertEqual(result.route, "simple")
        self.assertTrue(result.can_pass_to_agent5)
        self.assertEqual(result.selected_recipe.name, "계란볶음밥")

    def test_missing_candidates_are_reported(self):
        result = route_recipe(
            {
                "available_ingredients": ["rice", "egg"],
                "recipe_type": "korean",
            }
        )

        self.assertEqual(result.route_message, "candidate_foods_required")
        self.assertFalse(result.can_pass_to_agent5)

    def test_korean_inputs_are_supported_as_plain_text(self):
        result = route_recipe(
            {
                "available_ingredients": ["밥", "계란", "간장"],
                "recipe_type": "한식",
                "candidate_foods": [korean_egg_fried_rice_candidate()],
                "food_directions": {
                    "difficulty": "easy",
                    "cooking_time_limit_minutes": 15,
                    "preferred_cooking_method": "팬 조리",
                },
            }
        )

        self.assertEqual(result.route, "simple")
        self.assertTrue(result.can_pass_to_agent5)
        self.assertEqual(result.selected_recipe.name, "계란볶음밥")
        self.assertEqual(result.ingredients_to_use, ["밥", "계란"])
        self.assertEqual(result.seasonings_to_use, ["간장"])

    def test_ingredient_info_seasonings_are_not_mixed_into_ingredients_to_use(self):
        result = route_recipe(
            {
                "available_ingredients": ["rice", "egg", "soy_sauce"],
                "ingredient_info": {
                    "main_ingredients": ["rice", "egg"],
                    "sub_ingredients": [],
                    "seasonings": ["soy_sauce"],
                },
                "recipe_type": "korean",
                "candidate_foods": [
                    {
                        "name": "egg_rice",
                        "recipe_type": "korean",
                        "core_ingredients": ["rice", "egg"],
                        "required_ingredients": ["rice", "egg", "soy_sauce"],
                        "optional_ingredients": [],
                        "seasonings": [],
                        "difficulty": "easy",
                        "cooking_time_minutes": 10,
                    }
                ],
            }
        )

        self.assertEqual(result.route, "can_cook")
        self.assertEqual(result.ingredients_to_use, ["rice", "egg"])
        self.assertEqual(result.seasonings_to_use, ["soy_sauce"])

    def test_core_ingredients_block_when_missing_even_if_required_is_available(self):
        result = route_recipe(
            {
                "available_ingredients": ["돼지고기", "고추장"],
                "recipe_type": "korean",
                "candidate_foods": [
                    {
                        "name": "김치찌개",
                        "recipe_type": "korean",
                        "core_ingredients": ["김치"],
                        "required_ingredients": ["돼지고기", "고추장"],
                        "optional_ingredients": [],
                        "seasonings": [],
                        "difficulty": "easy",
                        "cooking_time_minutes": 10,
                    }
                ],
            }
        )

        self.assertEqual(result.route, "no_ingredient")
        self.assertFalse(result.can_pass_to_agent5)
        self.assertEqual(result.additional_ingredients, ["김치"])

    def test_route_recipe_node_returns_graph_friendly_dict(self):
        result = route_recipe_node(
            {
                "available_ingredients": ["rice", "egg", "soy_sauce"],
                "recipe_type": "korean",
                "candidate_foods": [egg_rice_candidate()],
                "food_directions": {
                    "difficulty": "easy",
                    "cooking_time_limit_minutes": 15,
                },
            }
        )

        self.assertIsInstance(result, dict)
        self.assertEqual(result["route"], "simple")
        self.assertEqual(result["selected_recipe"]["name"], "egg_rice")


if __name__ == "__main__":
    unittest.main()
