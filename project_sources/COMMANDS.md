# Command And Interaction Reference

Date: 2026-06-06

This project is not centered on a natural-language text parser. Player input is
handled through Textual focus/keys or legacy CLI aliases, then mapped to
structured dataclass commands in `src/game/app/commands.py`.

## Command Flow

1. UI displays `ScreenAction` choices from `ActionProvider`.
2. Player chooses by focus, Enter, number key, hotkey, or typed alias.
3. UI maps the choice to a command dataclass.
4. `AppController.handle(command)` dispatches to a flow or save/load.
5. The flow returns a `Result` with state, structured events, errors, and HCI
   analysis.

## Main Menu / Town Aliases

The legacy CLI accepts direct aliases such as:

- `start` / `new`
- `roster` / `r`
- `supplies` / `s`
- `ledger`
- `expedition` / `x`
- `save`
- `load`
- `help` / `?`
- `quit` / `q`

The Textual TUI uses focus, Enter, number keys, visible hotkeys, Esc/Backspace
for Back/Cancel, and fixed screen zones.

## Command Dataclasses

Company and town:

- `StartNewCompany(name="Haven Charter")`
- `ViewRoster()`
- `ViewSupplies()`
- `ViewGear()`
- `ViewLedger()`
- `ViewTown()`
- `ViewWorld()`
- `TravelWorld(destination_id)`
- `ViewMemorial()`
- `GenerateRecruitOffers()`
- `HireRecruit(offer_index)`
- `RecoverCompany()`
- `BuySupply(supply_id, quantity=1)`
- `PurchaseUpgrade(upgrade_id)`
- `PurchaseGear(gear_id)`
- `EquipGear(hero_id, gear_id)`
- `UnequipGear(hero_id)`
- `AssignActiveHero(hero_id, slot)`
- `AcceptContract(contract_id)`

Expedition and dungeon:

- `StartExpedition(expedition_id="opening", enter_maze=False,
  stop_at_breach=False, manual_combat=False, interactive_dungeon=False,
  use_known_route=True)`
- `TakeExpeditionChoice(choice_id)`
- `ViewDungeon()`
- `InspectDungeonRoom()`
- `MoveDungeon(node_id)`
- `UseDungeonAction(action_id)`
- `EnterGeneratedMaze(seed=None)`
- `RetraceGeneratedMaze()`
- `WithdrawGeneratedMaze()`
- `RetreatGeneratedMaze()`
- `ReturnFromDungeon()`
- `ViewExpeditionReport()`
- `ClearExpeditionReport()`

Manual combat:

- `StartManualCombat(encounter_id)`
- `ViewCombat()`
- `ChooseCombatSkill(skill_id)`
- `ChooseCombatTarget(target_id)`
- `ResolveCombatAction(skill_id, target_id)`
- `MoveCombatActor(to_slot)`
- `PassCombatTurn()`
- `DelayCombatTurn()`
- `RetreatCombat()`
- `ResolveCombatReaction(reaction_id)`
- `UseCombatSkill(actor_id, skill_id, target_id)`
- `SelectTarget(target_id)`
- `Retreat()`

System:

- `SaveGame(path)`
- `LoadGame(path)`
- `Quit()`

## ScreenAction Metadata

`ScreenAction` is the canonical UI action record:

```python
@dataclass(frozen=True)
class ScreenAction:
    number: str
    label: str
    value: str
    aliases: tuple[str, ...] = ()
    enabled: bool = True
    default: bool = False
    description: str = ""
    kind: ScreenActionKind | str = ScreenActionKind.GENERAL
    risk: ScreenActionRisk | str = ScreenActionRisk.SAFE
    cost: str = ""
    unavailable_reason: str = ""
    preview: str = ""
    result_hint: str = ""
    confirm: str = ""
```

Kinds include general, navigate, inspect, travel, town, combat, dungeon, system,
confirm, and cancel. Risks include safe, low, costly, risky, and irreversible.

## Combat Action Semantics

Manual combat-lite currently supports:

- choosing legal hero damage/support skills exposed by app combat views
- choosing legal targets based on combat rules
- moving a hero to adjacent 2x2 formation slots
- delaying, passing, and retreating where supported
- enemy automatic resolution
- reaction prompts where authored/available
- hit chance, damage estimate, damage label, and targeting reason as rule-derived
  previews outside UI rendering

It does not currently support full manual enemy control, arbitrary natural
language actions, mid-combat saves, or a broad support/movement/retreat system
beyond the implemented command set.

## Important Normalization Rule

When adding a new interaction, prefer this shape:

```text
input affordance -> ScreenAction -> command dataclass -> AppController flow ->
engine rule/result -> structured GameEvent -> view model -> UI render
```

Avoid adding direct UI mutations of `CompanyState` or combat state.
