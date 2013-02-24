% Example of use of kNN search CUDA.

% Parameters
ref_nb     = 4096;  % Reference point number, max=65535
query_nb   = 4096;  % Query point number,     max=65535
dim        = 32;    % Dimension of points
k          = 20;    % Nearest neighbors to consider
iterations = 100;

% Initialize reference and query points
ref   = single(rand(ref_nb,dim));
query = single(rand(query_nb,dim));

% Display informations
fprintf('Number of reference points      : %6d\n', ref_nb  );
fprintf('Number of query points          : %6d\n', query_nb);
fprintf('Dimension of points             : %4d\n', dim     );
fprintf('Number of neighbors to consider : %4d\n', k       );
fprintf('Processing kNN search           :'                );

% Call kNN search CUDA
tic
for i=1:iterations
    [dist, ind] = knn_cuda_with_indexes(ref,query,k);
end
elapsed_time = toc;
fprintf(' done in %f s for %d iterations (%f s by iteration)\n', elapsed_time, iterations, elapsed_time/iterations);
