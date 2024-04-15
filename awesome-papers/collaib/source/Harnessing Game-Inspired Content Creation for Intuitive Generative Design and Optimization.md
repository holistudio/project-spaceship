
**Link:** https://link.springer.com/chapter/10.1007/978-3-031-13249-0_13

**Authors**: [[Villagi, Lorenzo]], [[Stoddart, James]] [[Gaier, Adam]]

## Look Into
 - 

## Problem

"MOO requires the definition of objectives to be minimized—design exploration calls for features to be explored"

"Though generative design is gaining broader adoption in the AEC industry, its impact is limited by the level of technical skill required to operate computational design tools and the challenge of building generalizable applications"

WFC "extracts local patterns from a sparse set of examples and transforms them into a set of local constraints...These constraints drive generation and ensure that every local patch of the output also exists in the set of input examples."


WFC can provide a way to make it easier for designers to teach AI "experienced designers can show and teach what good designs look like and have the computer replicate virtually infinite variations of the provided examples...guided through exposed normalized weights assigned to each individual tile."




### Specific Challenges

Limitations of current WFC approaches
- WFC does not offer control over global constraints.
- Constraints are purely spatial (adjacency).
- Lacks input controls for a search algorithm.
- Lacks domain specific constraints.

## Related Works

 - ...
 - ...

## Approach

**Dataset:** ...

Quality-Diversity algorithms produce results "organized by high-level features better suited to the judgement of domain experts."


"varied qualities, such as the perimeter of a building or the number of bedrooms in a unit...produces a grid or ‘map’ of the solutions – with each axis corresponding to a feature."


A better WFC approach:
- Control of formal massing via global performance metrics (e.g., natural ventilation and noise) and global geometric features (e.g., building façade area) via integration with a QD optimization framework.
- Dynamic weighting for tile unit selection as optimization controls.
- Dynamic pre-constraining of tiles for improved searchability.
- Fixed pre-constraining with boundary solution tiles for design-domain ease of use.

Tiles catalog:
 - facade : mid-block or corner outer or corner inner
 - apartment: mid-block or corner (includes corridor space)
 - stair core
 - empty tile

Manually create a few good designs that demonstrate variations and desired adjacency rules

tile probability weights - the probability it will appear in the WFC results/output

variable tile pre-constraints - locked in, always appear in the WFC results

### Objective Function
- Indoor ventilation: connectivity distance of each room to the apartment’s windows
- Landscape Capacity for Carbon Sequestration: the amount of clear space green areas have from adjacent buildings
- Site noise: highway and road noise throughout the site using surrogate noise simulation models
- Number of apartment units
- Proximity of apartment to building core: must be less than 5 tiles

### QD Feature Space
- Facade Length
- Number of buildings
- Open area between buildings: number of empty tiles except those wedged b/w two buildings

### QD Algorithm
MAP-Elites - optimize an encoding composed of:
- vector of tile weights
- set of fixed tiles that children designs may inherit from the parent

T-DominO variation of MAP-Elits handles the multiple objectives and constraints:

"rewarding solutions with balanced performance over those which excel at only a single objective. Solutions which follow constraints are always preferred over those which do not."

Divide the feature sapce into discrete bins, each bin holding a single individual.

New solution is assigned a bin based on its features and whether its objective function fitness is higher than a solution that may already be in the bin.

In the end, each bin contains the best, the elite solution for that feature combination

QD allows the design space to be viewed through the lens of these features
## Key Findings

 - ...
 - ...
 - ...

## Takeaways

 - ...
 - ...
 - ..





