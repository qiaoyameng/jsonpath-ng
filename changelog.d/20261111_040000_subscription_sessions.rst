Added
-----

*   Add incremental subscription sessions in ``jsonpath_ng.subscription``.

    A ``SubscriptionDocument`` accepts versioned ``add``/``remove``/
    ``replace``/``move`` patches and maintains the hit sets of a bounded
    number of ``SubscriptionSession`` objects.  Subscriptions covering
    fields, array indices, wildcards, slices and scalar filters are updated
    from their recorded dependencies; expressions that cannot be maintained
    locally are recomputed explicitly.  Bad base versions and illegal patch
    paths leave the document and every subscription unchanged.
