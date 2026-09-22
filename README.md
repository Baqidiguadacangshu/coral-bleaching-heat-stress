## Coral Bleaching Heat Stress

<!-- This is the SD5913 assignment 2 template. Everything in this file is yours to
replace, and the check counts words: comments like this one are not words, so
delete each one as you write. Start with the heading: name the phenomenon.

Then, in this order, at least 150 words in total.

New to folders, paths, or the files here whose names start with a dot? Read
https://github.com/sd5913/pfad/blob/2026/reference/files.md first. Ten minutes. -->

![Coral bleaching heat stress animation](out/coral-bleaching-animation.gif)

## The phenomenon

Coral bleaching happens when corals experience environmental stress, especially unusually high ocean temperatures. Under prolonged heat stress, corals lose the symbiotic algae that provide much of their color and energy, causing them to become pale or white.

I chose coral bleaching heat stress because it is a natural phenomenon that changes over time but is difficult to understand from numbers alone. I wanted to explore how these changes could be translated into a visual form, where increasing heat stress gradually changes the color, density, and movement of a coral-like structure.--What goes up and down, and  why you looked at it. 

## The source

 The data comes from NOAA Coral Reef Watch's Bleaching Heat Stress Extent product. The CSV contains 14,789 daily records, with each row representing one date. It includes the global, Pacific, Atlantic, and Indian Ocean bleaching heat stress extent, recorded as both pixel counts (#) and proportions (%).

Source: https://coralreefwatch.noaa.gov/product/5km/index_5km_baa.php

## What the picture shows

The animation transforms NOAA bleaching heat stress data into a continuously changing coral-like form. Each year is represented by the changing color, density, movement, and particles of the structure: lower heat stress appears as a softer coral red, while higher stress gradually removes the color and pushes the form toward pale white.

To make the long-term change easier to see, I aggregated the daily NOAA data into yearly values and normalized them for the visual mapping. This removes daily and seasonal fluctuations, so the animation emphasizes the overall change through time rather than showing every individual measurement.

## Run it

```
uv run fetch.py
uv run plot.py
```
