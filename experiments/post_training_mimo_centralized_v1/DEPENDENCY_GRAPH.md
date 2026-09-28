# Dependencies

```mermaid
flowchart TD
  Source[Exact Titan / tokenizer / source identity] --> E0[Native E0 and resources]
  Split[Parent splits before generation] --> Verify[Verifier / tokens / provenance]
  Replay[Independent replay and preservation panels] --> E1
  E0 --> E1[Seed101 SFT / DEV screen]
  Verify --> E1
  E1 --> Seeds[Conditional seeds211/307]
  Seeds --> Check[Frozen endpoints / CHECK]
  Check --> Factors[Curriculum / preservation / scope / WSD]
  Factors --> Signal[Verified online successes]
  Signal --> Objectives[RSFT / GRPO / MaxRL]
  Objectives --> Sampling[Fixed vs NGU / adaptive sampling]
  Factors --> Skills[Independently competent donors]
  Skills --> Compose[Composition / distillation]
  Compose --> MiMo[Qualified deterministic MiMo sandbox]
  MiMo --> Confirm[Fresh confirmation / scale]
  Source --> Side[Independent architecture and inference controls]
```

GFlowRL additionally requires canonical reward-maximizing R0, reference/behavior
likelihood and reward support. LoopSpec additionally requires a useful looped model.
No side-track changes enter E1. Each future experiment receives a complete frozen
registration before its own native run; names here are not launch authorizations.
