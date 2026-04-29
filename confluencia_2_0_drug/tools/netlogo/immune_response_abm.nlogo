; Confluencia 2.0 simplified immune ABM (.nlogo)
; You can open this file directly in NetLogo.

extensions [csv]

breed [apcs apc]
breed [t-cells t-cell]
breed [b-cells b-cell]
breed [antibodies antibody]
breed [antigens antigen]

globals [
  antigen-pool
  cytokine-level
  activated-t-count
  plasma-b-count
  antibody-titer
  trigger-events
  selected-sample-id
]

apcs-own [presenting-level]
t-cells-own [active?]
b-cells-own [active? plasma?]
antibodies-own [strength]
antigens-own [potency]

to setup
  clear-all
  set-default-shape apcs "person"
  set-default-shape t-cells "person"
  set-default-shape b-cells "person"
  set-default-shape antibodies "dot"
  set-default-shape antigens "circle"

  set antigen-pool 0
  set cytokine-level 0
  set activated-t-count 0
  set plasma-b-count 0
  set antibody-titer 0
  set trigger-events []
  set selected-sample-id 0

  create-apcs 50 [
    set color orange
    set size 1.2
    set presenting-level 0
    setxy random-xcor random-ycor
  ]

  create-t-cells 140 [
    set color sky
    set size 1.0
    set active? false
    setxy random-xcor random-ycor
  ]

  create-b-cells 110 [
    set color violet
    set size 1.0
    set active? false
    set plasma? false
    setxy random-xcor random-ycor
  ]

  reset-ticks
end

to load-trigger-events [csv-path sample-id]
  let rows csv:from-file csv-path
  if length rows <= 1 [
    set trigger-events []
    stop
  ]

  let body but-first rows
  set trigger-events []
  set selected-sample-id sample-id

  foreach body [ r ->
    if length r >= 5 [
      let sid read-from-string item 0 r
      if sid = sample-id [
        let tk read-from-string item 1 r
        let imm read-from-string item 3 r
        let ag read-from-string item 4 r
        set trigger-events lput (list tk imm ag) trigger-events
      ]
    ]
  ]
end

to inject-antigen-from-triggers
  let now ticks
  let todays-events filter [ev -> item 0 ev = now] trigger-events
  if any? todays-events [
    foreach todays-events [ ev ->
      let immunogenicity item 1 ev
      let antigen-in item 2 ev
      set antigen-pool antigen-pool + antigen-in

      let spawn-count max list 1 round (antigen-in / 2)
      create-antigens spawn-count [
        set color red
        set size 0.8 + 0.5 * immunogenicity
        set potency immunogenicity
        setxy random-xcor random-ycor
      ]
    ]
  ]
end

to step-apc
  ask apcs [
    rt random 40 - random 40
    fd 0.5
    let local-antigen one-of antigens in-radius 1.5
    if local-antigen != nobody [
      set presenting-level presenting-level + [potency] of local-antigen
      ask local-antigen [ die ]
    ]
    set presenting-level max list 0 (presenting-level - 0.03)
  ]
end

to step-t-cell
  ask t-cells [
    rt random 30 - random 30
    fd 0.6
    let nearby-apc one-of apcs in-radius 1.2 with [presenting-level > 0.2]
    if nearby-apc != nobody [
      set active? true
      set color cyan
    ]

    if active? [
      set cytokine-level cytokine-level + 0.015
      if random-float 1 < 0.03 [ hatch 1 [ set active? true set color cyan ] ]
    ]
  ]
end

to step-b-cell
  ask b-cells [
    rt random 25 - random 25
    fd 0.5

    let active-t one-of t-cells in-radius 1.5 with [active?]
    if active-t != nobody [
      set active? true
      if random-float 1 < 0.05 [
        set plasma? true
        set color blue
      ]
    ]

    if plasma? [
      if random-float 1 < 0.2 [
        hatch-antibodies 1 [
          set color green
          set size 0.4
          set strength 0.8
          rt random 360
          fd 0.2
        ]
      ]
    ]
  ]
end

to step-antibody
  ask antibodies [
    rt random 35 - random 35
    fd 0.8

    let near-antigen one-of antigens in-radius 1.0
    if near-antigen != nobody [
      set antigen-pool max list 0 (antigen-pool - 1.0 * strength)
      ask near-antigen [ die ]
      if random-float 1 < 0.6 [ die ]
    ]

    if random-float 1 < 0.03 [ die ]
  ]
end

to decay-antigen
  ask antigens [
    if random-float 1 < 0.05 [ die ]
  ]
  set antigen-pool max list 0 (antigen-pool - 0.08 * antigen-pool)
  set cytokine-level max list 0 (cytokine-level - 0.02)
end

to update-readouts
  set activated-t-count count t-cells with [active?]
  set plasma-b-count count b-cells with [plasma?]
  set antibody-titer count antibodies
end

to go
  inject-antigen-from-triggers
  step-apc
  step-t-cell
  step-b-cell
  step-antibody
  decay-antigen
  update-readouts
  tick
end
@#$#@#$#@
GRAPHICS-WINDOW
200
10
736
547
-1
-1
10.4
1
10
1
1
1
0
0
0
1
-25
25
-25
25
1
1
1
ticks

BUTTON
16
18
92
51
setup
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
103
18
178
51
go
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

INPUTBOX
15
61
178
91
trigger-file
D:/IGEMji-cheng-fang-an/confluencia-2.0-drug/logs/epitope_triggers.csv
1
0
String

SLIDER
15
98
178
131
sample-id
sample-id
0
500
0
1
1
NIL
HORIZONTAL

BUTTON
15
138
178
170
load triggers
load-trigger-events trigger-file sample-id
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

MONITOR
15
183
178
228
antigen-pool
antigen-pool
3
1
11

MONITOR
15
232
178
277
activated-t-count
activated-t-count
0
1
11

MONITOR
15
281
178
326
plasma-b-count
plasma-b-count
0
1
11

MONITOR
15
330
178
375
antibody-titer
antibody-titer
0
1
11

PLOT
744
11
1114
190
Immune dynamics
Time
Count
0.0
100.0
0.0
500.0
true
false
"set-current-plot-pen \"T\"" "plot activated-t-count"
"set-current-plot-pen \"B\"" "plot plasma-b-count"
"set-current-plot-pen \"Ab\"" "plot antibody-titer"

PLOT
744
197
1114
376
Antigen and cytokine
Time
Level
0.0
100.0
0.0
400.0
true
false
"set-current-plot-pen \"antigen\"" "plot antigen-pool"
"set-current-plot-pen \"cytokine\"" "plot cytokine-level"

TEXTBOX
14
383
185
448
1) setup\n2) load triggers\n3) go (forever)
11
0.0
true
@#$#@#$#@
## WHAT IS IT?
This model is a simplified immune ABM with APC, T-cell, B-cell and antibody agents.

## HOW TO USE IT
1. Export trigger CSV from Python pipeline.
2. Set trigger-file and sample-id.
3. Click setup, then load triggers, then go.

## INPUT CSV
Header row required:
sample_id,tick,epitope_seq,immunogenicity,antigen_input

## OUTPUT SIGNALS
- antigen-pool
- activated-t-count
- plasma-b-count
- antibody-titer
- cytokine-level
@#$#@#$#@
NetLogo 6.4.0
@#$#@#$#@
