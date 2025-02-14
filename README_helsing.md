# Helsing Hackathon Challenge

# Challenges in Electronic Warfare: Detecting Radar Systems

## Overview

### Problem

Radar systems send out electromagnetic waves and listen to the reflected waves to scan their environment. We therefore refer to such systems as "emitters". The waves sent out by emitters can be analyzed to obtain information about the emitters themselves. This is done by capturing the waves and running algorithms and AI models against them in systems we'll call "sensors". In a scenario where we have a moving sensor, one of the (many) problems faced when analyzing captured signals is to find the precise geolocation of emitters and track them as the sensor moves.

### Objectives

To lessen the complexity we will provide datasets of so called PDWs (Pulse Descriptor Words), this means you do not have to work on the actual RF-data and instead you get descriptions of pulses with their characteristics.
These characteristics include thing like the time of arrival or the observed frequency of the pulse as well as the location information of the sensor at that point in time. 

1. Geolocation of the emitters
2. Tracking of emitters
    - Changes in parameters as well as location

## Getting Started

### Prerequisites

- Ability to parse Arrow IPC files

## Usage (Example)

    import polars as pl
    def load_window(file: Path) -> pl.DataFrame:
        df = pl.read_ipc(file)
        return df

This gets you started. From there explore the data. It will contain the ground truth data, you will know which Emitter each PDW came from and where it is located. 
