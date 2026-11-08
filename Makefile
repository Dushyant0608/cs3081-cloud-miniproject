.PHONY: train

# make train               -> full run (first 1500 records), saves to models_local/v1
# make train SUBSET=50     -> quick correctness check on 50 records only, no model saved to models/v1
train:
	python training/train.py --config training/config.yaml --version v1 $(if $(SUBSET),--subset $(SUBSET),)
