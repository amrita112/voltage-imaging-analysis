function bpod_data_struct = get_bpod_info(bpod_file_path, bpod_data_save_path)

bpod_data = load(bpod_file_path);
bpod_data = bpod_data.SessionData;

n_trials = bpod_data.nTrials;
left_right = bpod_data.TrialTypes;
cor_inc = zeros(n_trials, 1);
early_lick_sample = zeros(n_trials, 1);
early_lick_delay = zeros(n_trials, 1);
sample_start = zeros(n_trials, 1);
sample_end = zeros(n_trials, 1);
go_cue_start = zeros(n_trials, 1);
go_cue_end = zeros(n_trials, 1);
trial_start_times = zeros(n_trials, 1);

for trial = 1:n_trials
    trial_data = bpod_data.RawEvents.Trial(trial);
    trial_data = trial_data{1};
    cor_inc(trial) = ~isnan(trial_data.States.Reward(1));
    try
        early_lick_sample(trial) = ~isnan(trial_data.States.EarlyLickSample(1));
    catch
        early_lick_sample(trial) = 0;
    end
    try
        early_lick_delay(trial) = ~isnan(trial_data.States.EarlyLickDelay(1));
    catch
        early_lick_delay(trial) = 0;
    end
    sample_start(trial) = trial_data.States.SampleOn1(1);
    sample_end(trial) = trial_data.States.SampleOn1(2);
    go_cue_start(trial) = trial_data.States.ResponseCue(1);
    go_cue_end(trial) = trial_data.States.ResponseCue(2);
    trial_start_times(trial) = bpod_data.TrialStartTimestamp(trial);
end
bpod_data_struct.left_right = left_right;
bpod_data_struct.cor_inc = cor_inc;
bpod_data_struct.early_lick_sample = early_lick_sample;
bpod_data_struct.early_lick_delay = early_lick_delay;
bpod_data_struct.sample_start = sample_start;
bpod_data_struct.sample_end = sample_end;
bpod_data_struct.go_cue_start = go_cue_start;
bpod_data_struct.go_cue_end = go_cue_end;
bpod_data_struct.trial_start_times = trial_start_times;

save(bpod_data_save_path, 'bpod_data_struct')

return 

