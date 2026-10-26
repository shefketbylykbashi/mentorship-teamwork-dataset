function model = run_matlab_isa(toolkit_dir, isa_input_dir, output_dir, seed, epsilon, bound_flag)
%RUN_MATLAB_ISA Build a reproducible, descriptive MTP instance space.
%   MODEL = RUN_MATLAB_ISA(TOOLKIT_DIR, ISA_INPUT_DIR, OUTPUT_DIR, SEED)
%   reads ISA_INPUT_DIR/metadata.csv, prepares a fresh OUTPUT_DIR, and runs
%   the official andremun/InstanceSpace MATLAB toolkit. Optional EPSILON
%   changes the relative good-performance threshold (default 0.05).
%   Optional BOUND_FLAG turns IQR feature clipping on (default false).
%
%   This adapter is pinned to InstanceSpace commit
%   5f8cb25682202a8243bd2a2481ab8feb40c8c4ce. A clean checkout of
%   that commit is required. The toolkit requires MATLAB R2025a or later
%   and the toolboxes listed in its README. Its license is PolyForm
%   Noncommercial 1.0.0; consult that license before use.
%
%   Example (from this repository's root in MATLAB):
%     addpath('isa');
%     run_matlab_isa('/path/to/InstanceSpace', ...
%         'isa/outputs/rizana_trial_01', ...
%         'isa/outputs/rizana_trial_01_matlab_e05', 2026, 0.05);
%
%   The primary run disables the toolkit's median +/- 5*IQR feature
%   clipping: a sparse nonzero feature can otherwise lose much of its
%   variation. Use a separate output directory with BOUND_FLAG=true for
%   the prespecified bounding sensitivity analysis. Box-Cox/Z-score
%   normalization stays enabled in both runs.
%
%   The input metadata must have one row per instance; unique `instances`
%   identifiers; at least three finite numeric `feature_*` columns; and
%   at least one finite, nonnegative numeric `algo_*` column. The optional
%   `source` column is passed to the toolkit. Each output directory must
%   be new or empty. The saved options.json and run_info.json record the
%   effective options and toolkit revision. No held-out evaluation is
%   implied by the fitted PYTHIA/TRACE plots.

    if nargin < 4 || isempty(seed)
        seed = 2026;
    end
    if nargin < 5 || isempty(epsilon)
        epsilon = 0.05;
    end
    if nargin < 6 || isempty(bound_flag)
        bound_flag = false;
    end
    validateattributes(seed, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'integer', 'nonnegative'}, ...
        mfilename, 'seed');
    validateattributes(epsilon, {'numeric'}, ...
        {'scalar', 'real', 'finite', '>', 0, '<', 1}, ...
        mfilename, 'epsilon');
    validateattributes(bound_flag, {'logical'}, {'scalar'}, ...
        mfilename, 'bound_flag');

    toolkit_dir = local_absolute_path(toolkit_dir, 'toolkit_dir');
    isa_input_dir = local_absolute_path(isa_input_dir, 'isa_input_dir');
    output_dir = local_absolute_path(output_dir, 'output_dir');
    if ~isfolder(toolkit_dir) || ~isfile(fullfile(toolkit_dir, 'InstanceSpace.m'))
        error('MTPISA:missingToolkit', ...
            'toolkit_dir must contain InstanceSpace.m: %s', toolkit_dir);
    end
    if ~isfolder(isa_input_dir)
        error('MTPISA:missingInput', 'ISA input directory not found: %s', isa_input_dir);
    end
    metadata_file = fullfile(isa_input_dir, 'metadata.csv');
    if ~isfile(metadata_file)
        error('MTPISA:missingMetadata', 'metadata.csv not found: %s', metadata_file);
    end

    % Check the exact external code before creating any output. A clean
    % checkout makes source-version provenance meaningful.
    expected_commit = '5f8cb25682202a8243bd2a2481ab8feb40c8c4ce';
    actual_commit = local_git(toolkit_dir, 'rev-parse HEAD');
    if ~strcmp(actual_commit, expected_commit)
        error('MTPISA:toolkitRevision', ...
            'Expected InstanceSpace commit %s; found %s.', ...
            expected_commit, actual_commit);
    end
    if ~isempty(local_git(toolkit_dir, 'status --porcelain --untracked-files=no'))
        error('MTPISA:modifiedToolkit', ...
            'The InstanceSpace checkout has modified tracked files. Use a clean pinned checkout.');
    end

    % Fail before the toolkit can silently discard or reinterpret rows.
    [instance_count, feature_count, algorithm_count] = ...
        local_check_metadata(metadata_file);

    if isfolder(output_dir)
        contents = dir(output_dir);
        contents = contents(~ismember({contents.name}, {'.', '..'}));
        if ~isempty(contents)
            error('MTPISA:outputNotEmpty', ...
                'Output directory is not empty: %s', output_dir);
        end
    else
        [made, why] = mkdir(output_dir);
        if ~made
            error('MTPISA:mkdirFailed', 'Cannot create %s: %s', output_dir, why);
        end
    end

    % The current toolkit class adds its own core/output/utils paths when
    % constructed. Add utils temporarily here to resolve ISAdefaults.
    addpath(toolkit_dir, '-begin');
    addpath(fullfile(toolkit_dir, 'utils'), '-begin');
    resolved_class = which('InstanceSpace');
    if ~strcmpi(local_absolute_path(resolved_class, 'resolved_class'), ...
            fullfile(toolkit_dir, 'InstanceSpace.m'))
        error('MTPISA:toolkitShadowed', ...
            'MATLAB resolved InstanceSpace from a different directory: %s', resolved_class);
    end

    opts = struct();
    opts.general.seed = double(seed);
    opts.general.parallel = false;
    opts.perf.MaxPerf = true;
    opts.perf.AbsPerf = false;
    opts.perf.epsilon = double(epsilon);
    opts.bound.flag = bound_flag;
    opts.sifted.flag = true;
    opts.sifted.K = 10;
    opts.pilot.dims = 2;
    opts.pilot.analytic = false;
    opts.pythia.classifier = 'knn';
    opts.pythia.tuning = 'sobol';
    opts.trace.method = 'trace3';
    opts.outputs.csv = true;
    opts.outputs.png = true;
    opts.outputs.web = false;
    opts = ISAdefaults(opts);  % Save all effective defaults for this revision.

    copy_ok = copyfile(metadata_file, fullfile(output_dir, 'metadata.csv'));
    if ~copy_ok
        error('MTPISA:copyFailed', 'Could not copy metadata.csv into %s', output_dir);
    end
    local_write_json(fullfile(output_dir, 'options.json'), opts);

    provenance = struct();
    provenance.input_metadata = metadata_file;
    provenance.toolkit_commit = actual_commit;
    provenance.matlab_version = version;
    provenance.seed = double(seed);
    provenance.epsilon = double(epsilon);
    provenance.bound_flag = bound_flag;
    provenance.n_instances = instance_count;
    provenance.n_features = feature_count;
    provenance.n_algorithms = algorithm_count;
    provenance.status = 'started';
    local_write_json(fullfile(output_dir, 'run_info.json'), provenance);

    fprintf('[MTP ISA] %d instances, %d features, %d algorithms; epsilon = %.3f; bound = %d.\n', ...
        instance_count, feature_count, algorithm_count, epsilon, bound_flag);
    fprintf('[MTP ISA] Results: %s\n', output_dir);
    try
        isa = InstanceSpace(output_dir);
        isa = isa.build();
        model = isa.getResults();
        provenance.status = 'complete';
        provenance.selected_features = model.data.featlabels;
        local_write_json(fullfile(output_dir, 'run_info.json'), provenance);
    catch ME
        provenance.status = 'failed';
        provenance.error_identifier = ME.identifier;
        provenance.error_message = ME.message;
        local_write_json(fullfile(output_dir, 'run_info.json'), provenance);
        rethrow(ME);
    end
end

function p = local_absolute_path(value, label)
    if ~(ischar(value) || (isstring(value) && isscalar(value)))
        error('MTPISA:invalidPath', '%s must be a path string.', label);
    end
    p = char(value);
    if isempty(p)
        error('MTPISA:invalidPath', '%s must not be empty.', label);
    end
    % R2025a MATLAB has a JVM. Canonical paths avoid false mismatch from
    % relative paths and make the run's provenance unambiguous.
    p = char(java.io.File(p).getCanonicalPath());
end

function output = local_git(repo_dir, args)
    if contains(repo_dir, '"')
        error('MTPISA:invalidPath', 'A toolkit path cannot contain double quotes.');
    end
    command = ['git -C "' repo_dir '" ' args];
    [status, output] = system(command);
    output = strtrim(output);
    if status ~= 0
        error('MTPISA:gitFailed', 'Cannot inspect toolkit revision: %s', output);
    end
end

function [n_instances, n_features, n_algorithms] = local_check_metadata(path)
    fid = fopen(path, 'r', 'n', 'UTF-8');
    if fid < 0
        error('MTPISA:metadataRead', 'Cannot read %s', path);
    end
    header_line = fgetl(fid);
    fclose(fid);
    if ~ischar(header_line)
        error('MTPISA:metadataEmpty', 'metadata.csv is empty.');
    end
    header_line = strrep(header_line, char(65279), '');
    headers = string(strsplit(header_line, ','));
    if numel(unique(headers)) ~= numel(headers)
        error('MTPISA:duplicateColumns', 'metadata.csv has duplicate column names.');
    end
    names = cellstr(headers);
    if sum(strcmp(names, 'instances')) ~= 1
        error('MTPISA:instancesColumn', 'metadata.csv needs exactly one instances column.');
    end
    is_feature = startsWith(headers, 'feature_');
    is_algorithm = startsWith(headers, 'algo_');
    is_known = is_feature | is_algorithm | headers == 'instances' | headers == 'source';
    if ~all(is_known)
        error('MTPISA:unexpectedColumns', ...
            'Only instances, source, feature_*, and algo_* columns are allowed.');
    end
    n_features = sum(is_feature);
    n_algorithms = sum(is_algorithm);
    if n_features < 3 || n_algorithms < 1
        error('MTPISA:tooFewColumns', ...
            'metadata.csv needs at least 3 features and 1 algorithm.');
    end

    T = readtable(path, 'VariableNamingRule', 'preserve');
    if ~isequal(T.Properties.VariableNames, names)
        error('MTPISA:renamedColumns', ...
            'MATLAB changed metadata.csv column names; check the CSV header.');
    end
    n_instances = height(T);
    if n_instances < 3
        error('MTPISA:tooFewInstances', ...
            'At least three instances are needed for a two-dimensional ISA.');
    end
    ids = string(T.instances);
    if any(ismissing(ids) | strlength(strtrim(ids)) == 0) || ...
            numel(unique(ids)) ~= n_instances
        error('MTPISA:invalidInstances', ...
            'instances must contain unique, nonempty identifiers.');
    end
    for j = find(is_feature | is_algorithm)
        col = T.(names{j});
        if ~isnumeric(col) || ~isreal(col) || ~isvector(col) || ...
                numel(col) ~= n_instances || any(~isfinite(col))
            error('MTPISA:invalidNumericColumn', ...
                '%s must contain one finite numeric value per instance.', names{j});
        end
        if is_algorithm(j) && any(col < 0)
            error('MTPISA:negativePerformance', ...
                '%s contains negative performance, which PRELIM rejects.', names{j});
        end
        if is_feature(j) && max(col) == min(col)
            error('MTPISA:constantFeature', ...
                '%s is constant and should be removed before SIFTED.', names{j});
        end
    end
end

function local_write_json(path, value)
    fid = fopen(path, 'w', 'n', 'UTF-8');
    if fid < 0
        error('MTPISA:writeFailed', 'Cannot write %s', path);
    end
    wrote = fprintf(fid, '%s\n', jsonencode(value));
    closed = fclose(fid);
    if wrote < 0 || closed ~= 0
        error('MTPISA:writeFailed', 'Could not finish writing %s', path);
    end
end
