function convert(pathin, pathout, baseidx, flag_sparse)
  data = load(pathin);
  a = data.Problem.A;

  if flag_sparse
    n = length(a.jc) - 1;
    m = length(a.ir);

    r = a.ir;
    c = zeros(m, 1);
    delta = diff(a.jc);

    j = 1;
    for i=1:n
      d = delta(i);
      c(j:j+d-1) = i;
      j = j+d;
    end
  else
    n = length(a);
    m = nnz(a);
    [r, c, ~] = find(a);
  end

  f = fopen(pathout, 'w');
  fwrite(f, [n m], 'int32');
  fwrite(f, r - baseidx, 'int32');
  fwrite(f, c - baseidx, 'int32');
  fclose(f);
end
