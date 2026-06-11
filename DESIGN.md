# Design

## Game Pillars

- The company is the protagonist.
- Recruits begin generic and become memorable through survival, wounds, and death.
- The game is hard by default: losses, retreat, and partial success are part of play.
- The Maze is an exploited frontier/resource, not a simple evil dungeon.
- Civilization pulls the Maze deeper into the world one profitable expedition at a time.
- Player retreat is based on judgment, not a global timer.
- Engine systems return structured events; UI renders those events.
- Rules are small, deterministic, and testable.
- Authored content lives in YAML whenever practical.

Core fantasy: "Generic bodies go in. Stories come out."

Current working theme: "The Maze does not invade the world. The world invites it in,
one profitable expedition at a time."

## Narrative Direction

The player manages a chartered expedition company, not a chosen hero. Recruits are
ordinary people: debtors, mercenaries, failed soldiers, thieves, farmers, scholars,
criminals, wanderers, and desperate workers. The game should remember them because
they survived, were injured, killed something they should not have survived, fled
at the right moment, or died in a way the company ledger cannot quite reduce to
cost.

Pandora's Maze is the current working name for the impossible place. The name may
change later. The Maze gives the world treasure, relics, weapons, research, jobs,
fame, and economic growth. That usefulness is the trap. Towns, nobles, scholars,
merchants, rival companies, and desperate workers keep interacting with it because
the short-term rewards are real.

The Maze becomes worse because people keep using it. Civilization does not simply
fight the Maze; it hires the company to map it, loot it, stabilize it, sell access
to it, and recover bodies from it.

## Act Structure

Act 1, Frontier Company: the player starts in Haven Town with a small charter.
Contracts are grounded frontier work: roads, caves, bandits, beasts, ruins, missing
travelers, small settlements, local monsters, and bad maps. Maze leaks appear as
escaped creatures, repeated routes, impossible rooms, sealed entrances, relics,
survivor stories, and bosses carrying Maze-tainted objects.

Act 2, The Open Maze: the player reaches the wealthy town with the deepest known
entrance. Its economy depends on permits, relic auctions, map brokers, recruit
offices, corpse retrieval, guild halls, nobles, scholars, merchants, and fake
safe-route maps. The town is morally compromised because it is dependent, not
because every person there is evil.

Act 3, The Anchored Maze: repeated extraction makes the Maze more stable and more
connected to the overworld. Routes appear where they should not. Monsters migrate.
Old areas change. The company goes deeper because it has veterans, maps, route
knowledge, reputation, and survival history, not because prophecy points at it.

## Difficulty Direction

Hard should mean strategically harsh rather than randomly unfair.

- The default assumption is that early recruits are replaceable on paper and
  irreplaceable after play makes them specific.
- Combat should threaten Downed states, Mortal Wounds, death, and Order loss often
  enough that clean victories feel earned.
- Preparation should matter: supplies, formation, recovery, reserves, and contract
  choice should change risk.
- Retreat should be a skill. A company that leaves with partial loot and living
  veterans made a real decision.
- Rewards should tempt overreach. The best-paying choices should often be the ones
  that make the Maze more connected to the world.
- Avoid difficulty that only comes from hidden dice spikes. Prefer visible danger,
  scarce recovery, ugly tradeoffs, and consequences that persist.

## Damage Range Language

Damage variance should add texture without becoming the main source of difficulty.
Players should be able to plan around the maximum visible damage, especially when
deciding whether to guard, retreat, spend Effort, or accept a boss telegraph.

- Fixed: life states, combat tags, protection, healing, and core tactical promises.
- X-1 to X+1: most normal attacks.
- X-1 to X: weak, rushed, or utility attacks.
- X to X+1: heavy, disciplined, boss, or high-floor attacks.
- X-2 to X+2: wild or unstable attacks only.

Use narrow ranges by default. Do not hide damage ranges from the player, and do
not turn normal attacks into broad dice-style swings.

## Prototype Continuity

This is a clean Python rewrite inspired by an older C++ text adventure prototype.
It preserves the useful concepts without directly porting the implementation:

- Central company state owns flags, roster, inventory, supplies, breaches, and progress.
- UI selections become structured commands.
- Menu/controller flow is separate from rules.
- Choices can depend on flags, inventory, party strain, and known breaches.
- Items, skills, equipment, and formations are data-driven.
- Runtime systems return structured results for the UI to render.

## Formation Rules

Combat uses four explicit party slots:

- `BACK_LEFT`
- `BACK_RIGHT`
- `FRONT_LEFT`
- `FRONT_RIGHT`

Frontliners protect the backliner in the same lane only. A backliner is protected
when the same-lane frontliner is alive, not Downed, not stunned, not knocked down,
and still in the front slot.

Backliners become exposed if the front slot is empty, the frontliner is Downed,
dead, stunned, knocked down, or movement breaks the lane.

Movement swaps adjacent orthogonal slots. Diagonal movement is not a default rule.

## Targeting Rules

- Melee targets frontliners and exposed backliners.
- Reach targets frontliners and same-lane backliners, even when protected.
- Ranged targets any living enemy, but protected backliners apply a cover penalty.
- Magic targets any living enemy and ignores formation protection.

## Downed, Mortal Wounds, and Death

Heroes at 0 HP become Downed. A Downed hero cannot act and does not protect a lane.
Damage to a Downed hero adds one Mortal Wound. Three Mortal Wounds kill the hero
permanently. Healing a Downed hero above 0 HP removes Downed.

Enemies die at 0 HP in v0.1.

## Order Rules

Order starts at 6. Bad events reduce it:

- A hero becomes Downed: -1 Order.
- A hero dies: -2 Order.
- Horror effects: -1 Order by default.

Order 3 or less applies a -1 party initiative penalty. Order 0 indicates
panic/forced-retreat risk, but v0.1 does not force a full retreat system.

## Effort and No MP

Effort is the only special-action resource in v0.1. Basic skills are free.
Special skills spend Effort. There are no cooldowns, MP pools, or complex stamina.

## Company And Town Loop

Coin is the Haven currency for recruiting, recovery, supplies, gear,
upgrades, and Deep Surgery. Reputation is a standing/progression signal earned
through contracts and reports rather than the currency spent at town services.
Town services are data-tuned through `data/town.yaml`: recruitment offers,
recovery, supply purchases, memorial viewing, and active-party formation. The
roster cap is six. Four active party slots enter expeditions; living roster
members outside those slots are reserves.

World scaffolding lives in `data/world.yaml`: the current Maze name, starting
settlement, difficulty profile, overworld/future Maze locations, contract records,
and rumors. The v0.1 opening route starts with `blackwood_road_charter` active and
can complete that contract by clearing Shallow Cave.

Recruit backgrounds and motives live in `data/recruits.yaml` and are copied onto
runtime heroes. They should stay grounded and practical: debt, exile, wage work,
failed service, legal trouble, ambition, or survival.

Expedition nodes can set story flags, reveal rumor/lore entries, and complete
contracts. Company saves preserve known lore, active contracts, completed contracts,
flags, breaches, roster state, wounds, inventory, supplies, and history.

Recovery restores HP and Effort and clears Downed from living heroes, but it does
not erase Mortal Wounds.

Deep Surgery is a separate town service for living heroes with at least one
Mortal Wound. It spends Coin to remove one Mortal Wound, marks the hero In
Surgery, and benches them from the active party until the next expedition report
finalizes. HP and Effort still use normal Recovery Ward rules.

## Manual Combat-Lite

Interactive expeditions pause on hero turns. The player chooses exposed legal,
affordable damage/support skills, legal targets, adjacent formation movement,
pass/delay, and retreat where the current encounter flow allows it. Enemies
resolve automatically. Manual combat-lite does not add full manual enemy
control, arbitrary actions, broad tactical positioning, or mid-combat saves. The
UI may preview existing rule outcomes such as hit chance, damage estimate, and
targeting reason, but previews must be derived from combat rules outside the UI
renderer.

## v0.1 Content Scope

The opening route is:

Old Road -> Blackwood Forest -> Shallow Cave -> Cave Mini Boss -> Maze Breach ->
optional Pandora's Maze Depth 1 -> Haven Town.

The Shallow Cave is a standalone dungeon with a mini boss. Defeating the boss
reveals `shallow_cave_breach`, a staging chamber that can lead into the Maze.

The first contract should read as hard frontier work rather than heroic destiny.
The route includes a Maze leak before the full Maze descent: repeated road logic,
a loop-touched creature, and a breach that the company can report or exploit.

## Design Exclusions

Do not add stress bars, affliction/virtue systems, torch meters, global danger
timers, full overworld systems, hamlet-building systems, cooldowns, MP, complex
stamina, complex crafting, SQL, web backends, ECS, procedural editors, or mid-combat
saves unless explicitly requested.

## Future Editability

The designer should be able to tweak class stats, skills, enemies, supplies,
expedition nodes, wound thresholds, Order penalties, reputation, recruitment,
recruit backgrounds, contracts, rumors, locations, loot, known breaches, and town
services through small data or rules changes.
